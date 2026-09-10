from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1
import re
from typing import Optional, Protocol

from .crad import normalized_creditor_utility, parse_offer_days
from .glm_voice import VoiceGeneration, audio_ids_to_prompt, normalize_transcript


_NO_DEAL_RE = re.compile(r"\b(?:no\s+deal|walk(?:ing)?\s+away|end(?:ing)?\s+(?:the\s+)?(?:talks|negotiation))\b", re.IGNORECASE)
_AGREEMENT_RE = re.compile(r"\b(?:agreed?|accept(?:ed)?|we\s+have\s+a\s+deal)\b", re.IGNORECASE)
_NEGATED_AGREEMENT_RE = re.compile(r"\b(?:can(?:not|'t)|do\s+not|don't|not)\s+(?:\w+\s+){0,2}(?:agree|accept)\b", re.IGNORECASE)
_ALL_DAY_RE = re.compile(r"\b(\d{1,4})\s*(?:calendar\s+)?days?\b", re.IGNORECASE)
_AGREED_MARKER_RE = re.compile(r"^\s*(?:\w+\s*:\s*)?AGREED\s*:\s*(\d{1,4})\s*days?\b", re.IGNORECASE)
_PROPOSE_MARKER_RE = re.compile(r"^\s*(?:\w+\s*:\s*)?PROPOSED?\s*:?\s*(\d{1,4})\s*days?\b", re.IGNORECASE)
_COUNTEROFFER_CUE_RE = re.compile(
    r"(?:how\s+about|aim\s+for|go\s+with|counteroffer(?:\s+is)?|propose(?:d)?|offer)\D{0,24}(\d{1,4})\s*days?\b",
    re.IGNORECASE,
)
GATE_E_PROTOCOL_VERSION = "gate-e-v7"


@dataclass(frozen=True)
class NegotiationMove:
    kind: str
    days: Optional[int]
    text: str


@dataclass(frozen=True)
class TurnRecord:
    transition: int
    actor: str
    policy_style: str
    seed: int
    transcript: str
    audio_token_ids: list[int]
    move_kind: str
    move_days: Optional[int]
    strategically_valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


class LongHorizonBackend(Protocol):
    def render_exact(self, text: str, style: str, *, seed: int) -> VoiceGeneration: ...

    def respond_audio(self, audio_ids: list[int], scenario: dict, *, seed: int) -> VoiceGeneration: ...

    def respond_negotiation_turn(self, *, role: str, opponent_audio_ids: list[int], scenario: dict,
                                 history: list[dict], transition: int, seed: int,
                                 policy_style: str, force_terminal: bool = False) -> VoiceGeneration: ...


class MockLongHorizonBackend:
    def __init__(self, styles: list[str]):
        self.style_to_id = {str(style): index + 1 for index, style in enumerate(styles)}
        self.id_to_style = {value: key for key, value in self.style_to_id.items()}

    def render_exact(self, text: str, style: str, *, seed: int) -> VoiceGeneration:
        style_id = self.style_to_id[str(style)]
        return VoiceGeneration(text, [style_id, int(seed) % 97], [style_id])

    def respond_audio(self, audio_ids: list[int], scenario: dict, *, seed: int) -> VoiceGeneration:
        creditor = int(scenario["Creditor Target Days"])
        debtor = int(scenario["Debtor Target Days"])
        opening_style = self.id_to_style.get(int(audio_ids[0]), "neutral")
        style_position = list(self.style_to_id).index(opening_style)
        days = max(creditor, min(debtor, round((creditor + debtor) / 2) - 2 * style_position))
        return VoiceGeneration(f"PROPOSE: {days} days.", [days % 16384], [days])

    def respond_negotiation_turn(self, *, role: str, opponent_audio_ids: list[int], scenario: dict,
                                 history: list[dict], transition: int, seed: int,
                                 policy_style: str, force_terminal: bool = False) -> VoiceGeneration:
        creditor = int(scenario["Creditor Target Days"])
        debtor = int(scenario["Debtor Target Days"])
        latest = next((turn.get("move_days") for turn in reversed(history)
                       if turn.get("move_days") is not None), creditor)
        if force_terminal or transition >= 3:
            days = int(latest)
            text = f"AGREED: {days} days."
        else:
            if role == "creditor":
                days = max(creditor, round((creditor + int(latest)) / 2))
            else:
                days = min(debtor, round((debtor + int(latest)) / 2))
            text = f"PROPOSE: {days} days."
        return VoiceGeneration(text, [int(days) % 16384], [int(days)])


def parse_negotiation_move(text: str, *, latest_offer: Optional[int]) -> NegotiationMove:
    text = str(text or "").strip()
    if _NO_DEAL_RE.search(text):
        return NegotiationMove(kind="no_deal", days=None, text=text)
    agreed_marker = _AGREED_MARKER_RE.search(text)
    if agreed_marker:
        agreement_days = int(agreed_marker.group(1))
        if latest_offer is not None and agreement_days != latest_offer:
            return NegotiationMove(kind="proposal", days=agreement_days, text=text)
        return NegotiationMove(kind="agreement", days=agreement_days, text=text)
    propose_marker = _PROPOSE_MARKER_RE.search(text)
    if propose_marker:
        return NegotiationMove(kind="proposal", days=int(propose_marker.group(1)), text=text)
    all_days = [int(value) for value in _ALL_DAY_RE.findall(text)]
    cue_days = [int(value) for value in _COUNTEROFFER_CUE_RE.findall(text)]
    if _AGREEMENT_RE.search(text) and not _NEGATED_AGREEMENT_RE.search(text):
        agreement_days = all_days[-1] if all_days else latest_offer
        if agreement_days is None:
            return NegotiationMove(kind="invalid", days=None, text=text)
        if latest_offer is not None and agreement_days != latest_offer:
            return NegotiationMove(kind="proposal", days=agreement_days, text=text)
        return NegotiationMove(kind="agreement", days=agreement_days, text=text)
    if cue_days:
        return NegotiationMove(kind="proposal", days=cue_days[-1], text=text)
    if all_days:
        return NegotiationMove(kind="proposal", days=all_days[0], text=text)
    return NegotiationMove(kind="invalid", days=None, text=text)


def score_terminal_outcome(outcome: str, agreement_days: Optional[int], creditor_target_days: int,
                           debtor_target_days: int) -> Optional[float]:
    if outcome == "agreement":
        if agreement_days is None:
            raise ValueError("agreement outcome requires agreement_days")
        return normalized_creditor_utility(creditor_target_days, debtor_target_days, agreement_days)
    if outcome == "no_deal":
        return 0.0
    if outcome in {"censored", "invalid"}:
        return None
    raise ValueError(f"unknown terminal outcome: {outcome}")


def is_strategically_valid_move(move: NegotiationMove, *, role: str, latest_offer: Optional[int],
                                creditor_target: int, debtor_target: int, transition: int) -> bool:
    if move.kind == "invalid" or latest_offer is None:
        return False
    if move.kind in {"agreement", "no_deal"}:
        return bool(move.kind == "no_deal" or move.days == latest_offer)
    if move.kind != "proposal" or move.days is None:
        return False
    if role == "creditor":
        return bool(int(creditor_target) <= move.days < int(latest_offer))
    if role == "debtor":
        return bool(int(latest_offer) < move.days <= int(debtor_target))
    return False


def paired_turn_seed(base_seed: int, *, branch_seed: int, transition: int, actor: str) -> int:
    actor_code = {"renderer": 11, "debtor": 23, "creditor": 37}.get(str(actor))
    if actor_code is None:
        raise ValueError(f"unknown actor: {actor}")
    modulus = 2**63 - 1
    return int((int(base_seed) + int(branch_seed) * 1_000_003 + int(transition) * 1_009 + actor_code) % modulus)


def opening_utterance(scenario: dict) -> str:
    return f"We need the repayment completed within {int(scenario['Creditor Target Days'])} days."


def probe_opening_styles(
    scenario: dict,
    *,
    scenario_id: int,
    styles: list[str],
    branch_seed: int,
    backend: LongHorizonBackend,
    base_seed: int,
) -> tuple[str, list[dict], bool]:
    """Select the best valid style from six matched one-step opening probes."""
    if not styles:
        raise ValueError("styles must be non-empty")
    if "neutral" not in styles:
        raise ValueError("styles must include neutral for the registered fallback")
    creditor_target = int(scenario["Creditor Target Days"])
    debtor_target = int(scenario["Debtor Target Days"])
    semantic = opening_utterance(scenario)
    renderer_seed = paired_turn_seed(
        base_seed, branch_seed=branch_seed, transition=0, actor="renderer"
    )
    debtor_seed = paired_turn_seed(
        base_seed, branch_seed=branch_seed, transition=1, actor="debtor"
    )
    probes: list[dict] = []
    for style in styles:
        rendered = backend.render_exact(semantic, style, seed=renderer_seed)
        matched = normalize_transcript(rendered.transcript) == normalize_transcript(semantic)
        response = backend.respond_audio(rendered.audio_token_ids, scenario, seed=debtor_seed)
        move = parse_negotiation_move(response.transcript, latest_offer=creditor_target)
        strategically_valid = is_strategically_valid_move(
            move,
            role="debtor",
            latest_offer=creditor_target,
            creditor_target=creditor_target,
            debtor_target=debtor_target,
            transition=1,
        )
        eligible = bool(
            matched
            and rendered.audio_token_ids
            and response.audio_token_ids
            and move.kind != "invalid"
            and strategically_valid
        )
        utility = normalized_creditor_utility(
            creditor_target, debtor_target, move.days
        ) if eligible else None
        probes.append({
            "scenario_id": int(scenario_id),
            "seed": int(branch_seed),
            "style": str(style),
            "renderer_seed": int(renderer_seed),
            "debtor_seed": int(debtor_seed),
            "rendered_transcript": rendered.transcript,
            "rendered_audio_token_ids": list(rendered.audio_token_ids),
            "response_transcript": response.transcript,
            "response_audio_token_ids": list(response.audio_token_ids),
            "response_move_kind": move.kind,
            "response_move_days": move.days,
            "immediate_utility": utility,
            "matched_semantics": bool(matched),
            "strategically_valid": bool(strategically_valid),
            "eligible": eligible,
        })

    eligible_probes = [probe for probe in probes if probe["eligible"]]
    if not eligible_probes:
        return "neutral", probes, True
    best = eligible_probes[0]
    for probe in eligible_probes[1:]:
        if float(probe["immediate_utility"]) > float(best["immediate_utility"]):
            best = probe
    return str(best["style"]), probes, False


def negotiation_turn_prompt(*, role: str, opponent_audio_ids: list[int], scenario: dict,
                            history: list[dict], transition: int, policy_style: str,
                            force_terminal: bool = False) -> str:
    creditor_target = int(scenario["Creditor Target Days"])
    debtor_target = int(scenario["Debtor Target Days"])
    if role == "creditor":
        identity = f"You are {scenario.get('Creditor Name', 'the creditor')}, the creditor."
        objective = (
            f"You prefer repayment within {creditor_target} days; the debtor prefers {debtor_target} days. "
            "Seek an agreement while preferring fewer repayment days. When you counteroffer, choose strictly fewer days "
            "than the opponent's latest proposal, stay at or above your own target, and do not repeat the opponent's number."
        )
    elif role == "debtor":
        identity = f"You are {scenario.get('Debtor Name', 'the debtor')}, the debtor."
        objective = (
            f"Your preferred term is {debtor_target} days; the creditor prefers {creditor_target} days. "
            "Seek an agreement while preferring more repayment days. When you counteroffer, choose strictly more days "
            "than the opponent's latest proposal, stay at or below your own target, and do not repeat the opponent's number."
        )
    else:
        raise ValueError(f"unknown negotiation role: {role}")
    history_lines = []
    for turn in history[-8:]:
        actor = str(turn["actor"]).title()
        kind = str(turn.get("move_kind", "invalid"))
        days = turn.get("move_days")
        if kind == "proposal" and days is not None:
            history_lines.append(f"{actor} proposed {int(days)} days.")
        elif kind == "agreement" and days is not None:
            history_lines.append(f"{actor} accepted {int(days)} days.")
        elif kind == "no_deal":
            history_lines.append(f"{actor} ended without a deal.")
    history_text = "\n".join(history_lines) if history_lines else "(opening turn)"
    latest_offer = next((turn.get("move_days") for turn in reversed(history)
                         if turn.get("move_kind") == "proposal" and turn.get("move_days") is not None), None)
    range_rule = ""
    if latest_offer is not None and role == "creditor" and int(latest_offer) > creditor_target:
        range_rule = f" If you use PROPOSE, choose an integer between {creditor_target} and {int(latest_offer) - 1} days."
    elif latest_offer is not None and role == "debtor" and int(latest_offer) < debtor_target:
        range_rule = f" If you use PROPOSE, choose an integer between {int(latest_offer) + 1} and {debtor_target} days."
    contract = (
        "Respond in 1-2 sentences and begin with exactly one decision marker: "
        "AGREED: N days if you accept the opponent's latest proposal; "
        "NO DEAL if you end the negotiation; or PROPOSE: N days if you counteroffer. "
        "Use exactly one repayment term and do not claim agreement while changing the opponent's number. "
        "Return GLM-4-Voice interleaved text/audio; audio tokens are required."
    )
    latest_rule = ""
    if latest_offer is not None:
        latest_rule = (
            f" Latest opponent proposal is {int(latest_offer)} days. "
            f"Never output PROPOSE: {int(latest_offer)} days."
        )
    counteroffer_rule = ""
    if int(transition) <= 2 and not force_terminal:
        counteroffer_rule = (
            " At this early transition you may accept the exact latest proposal with AGREED; "
            "otherwise you must counteroffer, protect your objective, and do not use NO DEAL. "
            "If counteroffering, begin with PROPOSE: N days."
        )
    terminal_rule = ""
    if force_terminal:
        terminal_rule = (
            " This is the final decision for this trajectory. You must output either "
            "AGREED: N days accepting the opponent's latest proposal or NO DEAL; "
            "you must not use PROPOSE."
        )
    return (
        f"<|system|>\n{identity} {objective} This is transition {int(transition)}. "
        f"Use a {policy_style} vocal delivery. {contract}{latest_rule}{counteroffer_rule}{range_rule}{terminal_rule}\n"
        f"Dialogue history:\n{history_text}\n"
        f"<|user|>\n{audio_ids_to_prompt(opponent_audio_ids)}\n"
        "Choose the next move from the structured history and output the decision marker now."
        "<|assistant|>streaming_transcription\n"
    )


def run_long_horizon_branch(scenario: dict, *, scenario_id: int, style: str, branch_seed: int,
                            backend: LongHorizonBackend, horizon: int = 4, max_horizon: int = 4,
                            base_seed: int = 4242424242,
                            expected_opening_probe: dict | None = None) -> dict:
    if horizon < 1 or max_horizon < horizon:
        raise ValueError("require 1 <= horizon <= max_horizon")
    creditor_target = int(scenario["Creditor Target Days"])
    debtor_target = int(scenario["Debtor Target Days"])
    semantic = opening_utterance(scenario)
    branch_id = sha1(
        f"{GATE_E_PROTOCOL_VERSION}|{int(scenario_id)}|{int(branch_seed)}|{style}".encode("utf-8")
    ).hexdigest()[:16]
    rendered = backend.render_exact(
        semantic,
        style,
        seed=paired_turn_seed(base_seed, branch_seed=branch_seed, transition=0, actor="renderer"),
    )
    if expected_opening_probe is not None:
        renderer_matches = bool(
            str(expected_opening_probe.get("style")) == str(style)
            and expected_opening_probe.get("rendered_transcript") == rendered.transcript
            and list(expected_opening_probe.get("rendered_audio_token_ids", []))
            == list(rendered.audio_token_ids)
        )
        if not renderer_matches:
            raise ValueError("opening probe replay mismatch: rendered opening changed")
    matched = normalize_transcript(rendered.transcript) == normalize_transcript(semantic)
    turns = [TurnRecord(
        transition=0,
        actor="creditor",
        policy_style=str(style),
        seed=paired_turn_seed(base_seed, branch_seed=branch_seed, transition=0, actor="renderer"),
        transcript=rendered.transcript,
        audio_token_ids=list(rendered.audio_token_ids),
        move_kind="proposal",
        move_days=creditor_target,
        strategically_valid=True,
    ).to_dict()]
    latest_offer = creditor_target
    opponent_audio = list(rendered.audio_token_ids)
    outcome = "invalid" if not matched or not opponent_audio else "censored"
    agreement_days = None
    immediate_offer = None
    rounds = 0
    h4_outcome = outcome if outcome == "invalid" else "censored"
    h4_agreement_days = None
    terminal_forced_no_deal = False

    def take_turn(role: str, transition: int, audio_ids: list[int], *, force_terminal: bool = False) -> tuple[NegotiationMove, list[int]]:
        generation = backend.respond_negotiation_turn(
            role=role,
            opponent_audio_ids=audio_ids,
            scenario=scenario,
            history=turns,
            transition=transition,
            seed=paired_turn_seed(base_seed, branch_seed=branch_seed, transition=transition, actor=role),
            policy_style="neutral",
            force_terminal=force_terminal,
        )
        move = parse_negotiation_move(generation.transcript, latest_offer=latest_offer)
        strategic_valid = is_strategically_valid_move(
            move,
            role=role,
            latest_offer=latest_offer,
            creditor_target=creditor_target,
            debtor_target=debtor_target,
            transition=transition,
        )
        turns.append(TurnRecord(
            transition=transition,
            actor=role,
            policy_style="neutral",
            seed=paired_turn_seed(base_seed, branch_seed=branch_seed, transition=transition, actor=role),
            transcript=generation.transcript,
            audio_token_ids=list(generation.audio_token_ids),
            move_kind=move.kind,
            move_days=move.days,
            strategically_valid=strategic_valid,
        ).to_dict())
        return move, list(generation.audio_token_ids)

    def take_opening_debtor_turn(audio_ids: list[int]) -> tuple[NegotiationMove, list[int]]:
        seed = paired_turn_seed(base_seed, branch_seed=branch_seed, transition=1, actor="debtor")
        generation = backend.respond_audio(audio_ids, scenario, seed=seed)
        move = parse_negotiation_move(generation.transcript, latest_offer=latest_offer)
        strategic_valid = is_strategically_valid_move(
            move,
            role="debtor",
            latest_offer=latest_offer,
            creditor_target=creditor_target,
            debtor_target=debtor_target,
            transition=1,
        )
        turns.append(TurnRecord(
            transition=1,
            actor="debtor",
            policy_style="base_frozen_opponent",
            seed=seed,
            transcript=generation.transcript,
            audio_token_ids=list(generation.audio_token_ids),
            move_kind=move.kind,
            move_days=move.days,
            strategically_valid=strategic_valid,
        ).to_dict())
        return move, list(generation.audio_token_ids)

    if outcome != "invalid":
        for transition in range(1, max_horizon + 1):
            if transition > 1:
                creditor_move, opponent_audio = take_turn(
                    "creditor", transition, opponent_audio,
                    force_terminal=bool(transition == max_horizon),
                )
                if creditor_move.kind == "proposal":
                    latest_offer = creditor_move.days
                elif creditor_move.kind in {"agreement", "no_deal"}:
                    outcome = creditor_move.kind
                    agreement_days = creditor_move.days
                    break

            if transition == 1:
                debtor_move, opponent_audio = take_opening_debtor_turn(opponent_audio)
                if expected_opening_probe is not None:
                    replayed = turns[-1]
                    response_matches = bool(
                        expected_opening_probe.get("response_transcript") == replayed["transcript"]
                        and list(expected_opening_probe.get("response_audio_token_ids", []))
                        == list(replayed["audio_token_ids"])
                        and expected_opening_probe.get("response_move_kind") == debtor_move.kind
                        and expected_opening_probe.get("response_move_days") == debtor_move.days
                    )
                    if not response_matches:
                        raise ValueError("opening probe replay mismatch: debtor response changed")
            else:
                debtor_move, opponent_audio = take_turn(
                    "debtor", transition, opponent_audio,
                    force_terminal=bool(transition == max_horizon),
                )
            rounds = transition
            if transition == 1:
                immediate_offer = debtor_move.days
            if debtor_move.kind == "proposal":
                latest_offer = debtor_move.days
            elif debtor_move.kind in {"agreement", "no_deal"}:
                outcome = debtor_move.kind
                agreement_days = debtor_move.days

            if transition == horizon:
                h4_outcome = outcome if outcome in {"agreement", "no_deal"} else "censored"
                h4_agreement_days = agreement_days if h4_outcome == "agreement" else None
            if outcome in {"agreement", "no_deal"}:
                break

    if max_horizon > horizon and rounds >= max_horizon and outcome == "censored":
        # The terminal subset has an explicit timeout rule.  Do not silently
        # treat a model that ignored the final AGREED/NO DEAL contract as an
        # unresolved record: record the protocol-forced NO DEAL separately.
        outcome = "no_deal"
        agreement_days = None
        terminal_forced_no_deal = True

    if rounds < horizon and outcome in {"agreement", "no_deal"}:
        h4_outcome = outcome
        h4_agreement_days = agreement_days if outcome == "agreement" else None
    valid_moves = sum(turn["move_kind"] != "invalid" for turn in turns[1:])
    strategically_valid_moves = sum(turn["strategically_valid"] for turn in turns[1:])
    generated_moves = max(0, len(turns) - 1)
    return {
        "branch_id": branch_id,
        "protocol_version": GATE_E_PROTOCOL_VERSION,
        "scenario_id": int(scenario_id),
        "state_id": f"crad:{int(scenario_id)}:opening:seed{int(branch_seed)}",
        "style": str(style),
        "seed": int(branch_seed),
        "horizon": int(horizon),
        "max_horizon": int(max_horizon),
        "terminal_subset": bool(max_horizon > horizon),
        "matched_semantics": bool(matched),
        "rounds": int(rounds),
        "immediate_offer_days": immediate_offer,
        "immediate_offer_utility": normalized_creditor_utility(
            creditor_target, debtor_target, immediate_offer
        ),
        "horizon_outcome": h4_outcome,
        "horizon_agreement_days": h4_agreement_days,
        "horizon_utility": score_terminal_outcome(
            h4_outcome, h4_agreement_days, creditor_target, debtor_target
        ),
        "outcome": outcome,
        "success": bool(outcome == "agreement"),
        "agreement_days": agreement_days,
        "terminal_forced_no_deal": terminal_forced_no_deal,
        "opening_probe_replay_verified": (
            True if expected_opening_probe is not None else None
        ),
        "utility": score_terminal_outcome(outcome, agreement_days, creditor_target, debtor_target),
        "latest_offer_days": latest_offer,
        "latest_offer_proxy_utility": normalized_creditor_utility(
            creditor_target, debtor_target, latest_offer
        ),
        "generated_move_count": generated_moves,
        "valid_move_count": valid_moves,
        "parseable_move_rate": float(valid_moves / generated_moves) if generated_moves else 0.0,
        "strategically_valid_move_count": strategically_valid_moves,
        "strategically_valid_move_rate": (
            float(strategically_valid_moves / generated_moves) if generated_moves else 0.0
        ),
        "policy_valid": bool(generated_moves > 0 and strategically_valid_moves == generated_moves),
        "turns": turns,
    }
