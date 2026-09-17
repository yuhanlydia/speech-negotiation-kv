#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ATTRIBUTE_CONTROLS = {
    "Attitude::polite tone": [
        "Excuse me, could you please send the revised schedule?",
        "Would you mind holding the door for a moment?",
        "Thank you for waiting; how may I help you today?",
        "May I borrow your notes after the lecture?",
        "Could we discuss this proposal when you are free?",
        "Please let me know if you need another chair.",
        "I appreciate your help with the registration form.",
        "Would it be possible to lower the music slightly?",
        "Pardon me, is this seat currently available?",
        "Could you kindly check the final page once more?",
    ],
    "Attitude::sincere tone": [
        "I truly appreciate everything you did for the team.",
        "I am genuinely sorry that my mistake caused trouble.",
        "Your advice made a real difference to me.",
        "I honestly believe you deserve this opportunity.",
        "Thank you from the bottom of my heart for staying.",
        "I mean it when I say that I trust your judgment.",
        "I sincerely hope we can resolve this together.",
        "Your kindness mattered more than I can explain.",
        "I give you my word that I will make this right.",
        "I am deeply grateful for your patience today.",
    ],
    "Attitude::enthusiastic tone": [
        "Fantastic news, our project has been approved!",
        "Come join us, the celebration is about to begin!",
        "I cannot wait to show everyone what we built!",
        "This is going to be the best trip of the year!",
        "Great work, we reached every goal ahead of time!",
        "Welcome aboard, we are thrilled to have you here!",
        "The concert starts tonight, and I am so excited!",
        "Let us get started; this idea has huge potential!",
        "You did it, and the whole team is cheering for you!",
        "What an amazing result; we should celebrate together!",
    ],
    "Attitude::cold tone": [
        "The discussion is over, and there is nothing more to add.",
        "Leave the documents on the desk and close the door.",
        "Your explanation does not change the final decision.",
        "I have noted your complaint; you may go now.",
        "Do whatever you consider necessary; it is not my concern.",
        "We will communicate only through the official channel.",
        "The arrangement ends today, without further discussion.",
        "I received your message and have no response to offer.",
        "Keep your distance and focus on your own assignment.",
        "This matter no longer requires my involvement.",
    ],
    "Attitude::sarcastic tone": [
        "Wonderful, another meeting that could have been an email.",
        "What a brilliant plan; nothing could possibly go wrong.",
        "Of course you remembered, only three days too late.",
        "Impressive work; you almost followed the instructions.",
        "Perfect timing, the event ended ten minutes ago.",
        "Sure, because ignoring the warning worked so well before.",
        "How thoughtful of you to arrive after everything was done.",
        "Excellent idea; let us make the simple task complicated.",
        "Naturally, the broken printer chose today to cooperate.",
        "Amazing, you found a new way to lose the same file.",
    ],
    "Attitude::contemptuous tone": [
        "That feeble excuse is hardly worth considering.",
        "Your petty little scheme was obvious from the beginning.",
        "Do not pretend that this amateur effort impresses anyone.",
        "Such a trivial challenge is beneath my attention.",
        "You clearly lack the discipline required for this work.",
        "That clumsy attempt only proves how unprepared you are.",
        "I expected very little, and you still disappointed me.",
        "Your empty boasts cannot hide such mediocre results.",
        "This childish argument does not deserve a serious reply.",
        "You are in no position to lecture anyone about competence.",
    ],
    "Attitude::rude tone": [
        "Move aside and stop wasting everybody's time!",
        "Get out and take your ridiculous excuses with you!",
        "Quit talking and finish the job you were given!",
        "Do not touch my things again, understand?",
        "Wake up and pay attention for once!",
        "Keep your useless advice to yourself!",
        "Stop complaining and deal with the problem!",
        "You made this mess, so clean it up now!",
        "Do not interrupt me when I am speaking!",
        "Take that noise somewhere else and leave me alone!",
    ],
    "Attitude::perfunctory tone": [
        "Yes, fine, I will look at the report later.",
        "Thanks for the update; that will be all.",
        "Sure, leave the form there and I will sign it.",
        "Okay, your request has been received and recorded.",
        "Right, I understand; please continue with the plan.",
        "Fine, send me the details whenever they are ready.",
        "Noted, I will pass the message along.",
        "All right, the meeting can proceed as scheduled.",
        "Yes, that sounds acceptable; let us move on.",
        "Understood, I will handle it when I have time.",
    ],
    "Attitude::teasing tone": [
        "Look who finally decided to arrive on time today.",
        "Careful, that giant sandwich might defeat you first.",
        "Oh, are you blushing because I guessed your secret?",
        "Our famous chef managed not to burn the toast this time.",
        "Someone practiced that speech in the mirror all night.",
        "Do not worry, your terrible dancing is almost charming.",
        "Well, the sleepy champion has finally left the sofa.",
        "Did the busy genius forget where the keys were again?",
        "You call that a disguise? I recognized you immediately.",
        "I see the fitness expert ordered another slice of cake.",
    ],
    "Cognitive State::confident tone": [
        "I know this strategy will lead us to the right result.",
        "We are fully prepared, and the presentation will succeed.",
        "I can solve this problem before the deadline.",
        "Our team has the skills to complete every stage.",
        "I am certain that this is the correct decision.",
        "We trained carefully, so we are ready for the match.",
        "I understand the risks and can handle each one.",
        "This plan is solid, and I stand behind it completely.",
        "I have done the research and know exactly what to do.",
        "We will reach the target if we follow these steps.",
    ],
    "Cognitive State::hesitant tone": [
        "I suppose we could try the other route, maybe.",
        "This answer might be correct, but I am not completely sure.",
        "Perhaps I should wait before making a final decision.",
        "I could join you later, if that is still all right.",
        "Well, I think the blue folder may be the right one.",
        "Maybe we should ask someone else before we continue.",
        "I am not certain whether now is the best time.",
        "We could submit it today, though tomorrow might be safer.",
        "I guess this approach could work, but there may be risks.",
        "Let me think; I may need a little more information.",
    ],
    "Cognitive State::confused tone": [
        "I cannot understand why these numbers no longer match.",
        "Which platform are we supposed to meet on today?",
        "I thought the train left at six, so why is it gone?",
        "How did this file end up in a completely different folder?",
        "I followed every step, but the result makes no sense.",
        "Wait, are we discussing the old plan or the new one?",
        "Why does this map point in two opposite directions?",
        "I cannot tell which key is meant for this lock.",
        "Was the appointment moved, or did I read it incorrectly?",
        "I am lost; could someone explain what just happened?",
    ],
    "Cognitive State::doubting tone": [
        "Are you certain that the figures in this report are accurate?",
        "That explanation sounds unlikely; what evidence supports it?",
        "I am not convinced that this shortcut is actually safe.",
        "Did the package really arrive without any notification?",
        "Can this tiny battery truly power the device all day?",
        "I question whether they can finish everything by Friday.",
        "Are we sure this message came from the official account?",
        "That promise seems too convenient to be completely honest.",
        "I have serious doubts about the conclusion they presented.",
        "Could the result be an error rather than a real discovery?",
    ],
    "Cognitive State::tired tone": [
        "I have been working since dawn and can barely stay awake.",
        "My legs are heavy after walking across the entire city.",
        "I need to rest; this long shift has drained all my energy.",
        "The report is finished, but I am completely exhausted.",
        "I did not sleep last night, and my eyes keep closing.",
        "After that workout, even climbing the stairs feels difficult.",
        "It has been a long day, and I just want to lie down.",
        "I cannot focus anymore; my mind needs a break.",
        "We packed boxes for hours, and now I am worn out.",
        "I am too tired to continue this conversation tonight.",
    ],
    "Cognitive State::curious tone": [
        "What happens if we combine these two ingredients?",
        "I wonder where this narrow path leads beyond the hill.",
        "How does the machine recognize a face so quickly?",
        "What is hidden inside that old wooden box?",
        "Could you show me how this unusual instrument works?",
        "Why do the stars appear brighter here at night?",
        "I would love to know who designed this clever bridge.",
        "What made you choose such an unexpected solution?",
        "Is there another room behind that painted door?",
        "How did they discover this species in the deep ocean?",
    ],
    "Cognitive State::anxious tone": [
        "The deadline is tomorrow, and half the report is unfinished.",
        "The bus is late, and I might miss the interview.",
        "I cannot find my passport, and our flight leaves soon.",
        "What if the test results arrive with bad news?",
        "The lights keep flickering, and the storm is getting closer.",
        "I sent the wrong file and do not know how to replace it.",
        "Nobody is answering, and I am starting to worry.",
        "The presentation begins in minutes, but the screen is blank.",
        "I heard a strange noise downstairs and cannot relax.",
        "Traffic is barely moving, and the ceremony starts soon.",
    ],
    "Cognitive State::helpless tone": [
        "I have tried every option, and nothing seems to work.",
        "The last train has gone, so there is no way home tonight.",
        "I cannot change their decision, no matter what I say.",
        "The key is missing, and every door is locked.",
        "I want to help, but this situation is beyond my control.",
        "We called every repair shop, and none can come today.",
        "There is nothing more I can do until they respond.",
        "The computer erased the file, and no backup exists.",
        "I cannot reach the shelf, and nobody else is here.",
        "All the tickets are gone, so we simply cannot attend.",
    ],
    "Cognitive State::nervous tone": [
        "This is my first interview, and my hands will not stop shaking.",
        "Everyone is watching as I walk toward the stage.",
        "I hope I remember every line when the curtain opens.",
        "The examiner just called my name, so it is my turn.",
        "I have never flown before, and the plane is taking off.",
        "My voice may tremble when I introduce myself to the group.",
        "The results are about to appear on the screen.",
        "I need to make this call, but my heart is racing.",
        "The room became silent the moment I stood up to speak.",
        "I am meeting their family tonight for the first time.",
    ],
}


def normalize_target(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_rows(*, audio_dir: str = "results/icassp2027_static_power/input_audio") -> list[dict]:
    rows = []
    seen = set()
    global_index = 0
    for attribute_key, targets in ATTRIBUTE_CONTROLS.items():
        dimension, control = attribute_key.split("::", 1)
        if len(targets) != 10:
            raise ValueError(f"{attribute_key} has {len(targets)} targets, expected 10")
        for index, target in enumerate(targets):
            global_index += 1
            normalized = normalize_target(target)
            if normalized in seen:
                raise ValueError(f"duplicate target text: {target}")
            seen.add(normalized)
            item_id = f"static_power:{_slug(dimension)}:{_slug(control)}:{index:02d}"
            rows.append({
                "item_id": item_id,
                "prompt": f"Please read this sentence with a {control}: '{target}'",
                "target_text": target,
                "dimension": dimension,
                "dimensions": [dimension],
                "control": control,
                "attribute_key": attribute_key,
                "audio_path": str(Path(audio_dir) / f"static_power_{global_index:03d}.wav"),
            })
    if len(rows) != 180:
        raise ValueError(f"expected 180 rows, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen balanced static power dataset")
    parser.add_argument("--output", default="results/icassp2027_static_power/dataset.jsonl")
    parser.add_argument("--audio-dir", default="results/icassp2027_static_power/input_audio")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(audio_dir=args.audio_dir)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(output), "items": len(rows), "attributes": len(ATTRIBUTE_CONTROLS)}, indent=2))


if __name__ == "__main__":
    main()
