# Evaluation Strategy and Plan

This document describes the **evaluation strategy** and the concrete
**evaluation plan** for the bilingual (English / Hungarian) voice
appointment chatbot.

The system under test is the web client of `vetbot-web`: a FastAPI +
React SPA where the user holds the microphone button (or `Enter`),
speaks an utterance, and the server runs an ASR → sentiment → dialogue
(Claude Haiku tool-use) → TTS pipeline to fill the booking slots of the
selected domain (currently *Paws & Whiskers* veterinary clinic and
*Studio Lumen* hair salon). A successful session ends with a JSON
appointment record under `output/`.

---

## 1. Evaluation Strategy

The strategy is the high-level "why and what" of the evaluation, before
any concrete test session is scheduled.

### 1.1 Why evaluate now?

- The voice-first, push-to-talk UX
  is brand-new and has never been observed in the hands of a user who
  is not the author.
- The pipeline mixes **two languages and code-switching** mid-session.
  Whisper, the XLM-RoBERTa sentiment model, the Haiku tool-use loop and
  the Piper voices all behave well in isolation, but their joint
  behaviour on real user speech has not been measured.
- The dialogue is driven by an LLM with **declared tools** rather than
  a hand-written state machine. The booking flow therefore has emergent
  branching that benefits from observation, not just unit tests.
- We want to compare two cohabiting domains (**vet** and
  **hairdresser**) to check that the domain-registry abstraction
  (`domains/*.yaml` + sibling Markdown) really lets a non-author
  understand both flows without per-domain instructions.

### 1.2 What do we want to learn? — Quesenberry's *Five Es*

The strategy measures each of the five usability criteria against
concrete questions about the chatbot:

| Criterion | Concrete question for this system |
| --- | --- |
| **Effective** | Does the session end with a JSON file that contains the *correct* name, phone number, pet/service, time, and complaint as actually spoken by the user? |
| **Efficient** | How long does a complete booking take (wall-clock and number of user turns), and how often does Whisper / Haiku need a clarification round? |
| **Engaging** | Do users find the synthesised voices natural enough, the slot panel reassuring, and the push-to-talk affordance pleasant — or does it feel like fighting the microphone? |
| **Error tolerant** | When the user mis-speaks, code-switches, gives a half-heard digit, or interrupts, does the bot recover (especially via the phone digit-readback and `confirm_phone` tool) instead of saving a corrupted booking? |
| **Easy to learn** | Can a first-time user finish a booking without the moderator explaining the UI — i.e., is "hold the button, speak, release" discoverable from the screen alone? |

### 1.3 What kind of data?

Quantitative — collected automatically by the server and timing tools:

- Wall-clock time per booking (first `utterance` event to `saved`
  event on the WebSocket).
- Number of user turns and number of `ask_kb` / clarification turns
  per booking.
- Slot accuracy: per slot, was the persisted value identical to the
  ground truth on the task card? (correct / corrected by user / wrong)
- Booking completion rate: bookings that reach a `saved` event vs.
  sessions that are abandoned.
- ASR word error rate, sampled by comparing the displayed `transcript`
  to a transcriptionist's gold for ~10 % of utterances.
- Audio I/O failure rate: number of utterances where the browser
  recorder, ffmpeg decode, or Piper playback failed.

Qualitative — collected by the moderator and from the participant:

- Think-aloud commentary during the session (audio + notes).
- Post-task SUS (System Usability Scale, 10 items, translated to
  Hungarian for HU participants).
- Three open questions after the session:
  1. What surprised you the most about the conversation?
  2. Was there a moment where you felt the bot misunderstood you, and
     what did you do?
  3. Would you trust this booking to actually appear in the clinic's
     calendar? Why or why not?

### 1.4 What version is under test?

- **Branch** `main` at the commit pinned at the start of the evaluation
  week (recorded in the test log).
- **Both domains** (`vet`, `hairdresser`) — selectable from the URL or
  the configuration; each participant tests both, order counter-balanced.
- **Web client** served by `vetbot-web` on `http://127.0.0.1:8000`,
  Chrome on Windows 11. (The CLI is *not* tested — the web client is
  the primary surface for the lab demo.)
- **CPU vs. GPU**: we run the GPU build (`uv sync --extra cuda`,
  Whisper `large-v3`, float16) because that is the configuration the
  grader will see at the demo. CPU mode (`small`, int8) is exercised
  only as a sanity check, not as the primary measurement.

### 1.5 Constraints

- **People**: 4–5 volunteer testers from the author's circle; one
  moderator (the author); one note-taker (best effort, may be the same
  person between sessions).
- **Cost**: zero monetary budget. Anthropic API tokens for Haiku 4.5
  are paid from the author's existing course allowance; prompt caching
  keeps the per-session cost in the cent range.
- **Time**: one calendar week including recruitment, sessions and
  write-up. Each session is capped at 60 minutes.
- **Hardware**: a single laptop with an NVIDIA GPU (RTX-class, 16 GB
  VRAM), a USB headset for the participant, and a backup wired
  microphone. Sessions are run on the author's premises (quiet room).
- **Language coverage**: at least two HU-native and two EN-comfortable
  participants so that both language paths are exercised.

---

## 2. Evaluation Plan

The plan is the concrete execution of the strategy: who, what, when,
where, how the data is collected, and what we do with it afterwards.

### 2.1 Who?

- **User population**: technically literate adults who have booked at
  least one appointment by phone or web in the last year. No prior
  exposure to this project.
- **Recruiting**: personal network of the author. Each participant is
  briefed by email with a one-paragraph description of the project and
  a consent form covering audio recording and anonymised note-taking.
- **Sample size**: **5 testers**. Nielsen's classic result (≈5 users
  uncover ≈80 % of usability issues) is appropriate for a course
  project; recruiting more would not change the conclusions but would
  exceed the time budget.
- **Language split**: 3 Hungarian-first sessions, 2 English-first
  sessions. Every participant additionally does at least one turn in
  the *other* language to exercise code-switching.
- **Moderators / observers**:
  - *Moderator* (author): runs the session, reads task cards, asks the
    think-aloud prompts, handles consent.
  - *Observer / note-taker*: tracks slot-by-slot success in a paper
    sheet and timestamps notable events. If no second person is
    available, the screen-capture + WebSocket log substitute for the
    observer.
- **Motivation**: testers are not paid. They are offered coffee, a
  printed summary of the result, and acknowledgement in the lab report
  (with their consent). The promise of a *short* (≤60 min) session is
  the main draw.

### 2.2 What?

- **Version**: `main` at the pinned commit. The exact SHA is written
  on the first row of the test log on the day of the first session.
- **Device**: the author's laptop, Chrome (latest stable), USB headset
  for the participant, external display so the moderator can see the
  slot panel without leaning over.
- **Two task cards per participant**, one per domain. The cards are
  printed on paper so the participant does not read from the same
  screen the chatbot is rendered on:

  **Task card V — Veterinary (Paws & Whiskers)**

  > Your dog *Bogi*, a 6-year-old Vizsla, has been limping on his
  > right front leg since this morning. Book the earliest possible
  > appointment for next week. Your name is *[participant's own name]*,
  > your phone number is *+36 30 555 0142*. If the bot offers vaccine
  > reminders or dental cleaning, decline politely.

  **Task card H — Hair salon (Studio Lumen)**

  > You want a *cut and blow-dry* on Friday afternoon, ideally between
  > 16:00 and 18:00. Your name is *[participant's own name]*, your
  > phone number is *+36 70 555 0381*. Ask the bot how long the service
  > takes before you confirm.

  Each card has the same shape (identification → service / complaint →
  time → confirmation) so that timings between the two domains are
  comparable, but the wording is deliberately *underspecified* in
  places to force the bot to ask follow-up questions.

- **Order**: the two cards are presented in randomised order across
  participants (Latin-square style with N=5: VHVHV / HVHVH / VHHVH…),
  with a 5-minute break between cards.
- **Language instructions on the card**: the card is written in the
  participant's preferred language but instructs them that they *may*
  switch language mid-session if they want to.

### 2.3 When?

- **Calendar window**: one week, 2026-05-18 → 2026-05-22 (Mon–Fri).
- **Per-session budget**:

  | Phase | Duration |
  | --- | --- |
  | Greeting, consent, recording setup | 5 min |
  | Briefing + microphone test utterance | 5 min |
  | Task V (or H — randomised) | 15 min |
  | Short break / SUS for that domain | 5 min |
  | Task H (or V) | 15 min |
  | SUS + 3 open questions + debrief | 10 min |
  | Buffer (technical hiccups, retries) | 5 min |
  | **Total** | **60 min** |

- **Slack**: two unscheduled 60-minute slots are kept open at the end
  of the week as fallbacks if a participant cancels or a session has
  to be re-run because of a recording failure.

### 2.4 Where?

- A quiet room in the author's flat with controllable lighting and no
  background TV/radio.
- The participant sits at the laptop with the USB headset; the
  moderator sits at a 90° angle so screen + face are both visible to
  the camera but the moderator is not in the participant's eye line.
- Only one participant in the room at a time; arrivals scheduled with
  ≥15 minutes between sessions so testers do not meet each other and
  contaminate expectations.

### 2.5 Full test procedure

For every session, in order:

1. **Setup (moderator, before participant arrives)**
   1. Pull latest `main`, confirm the pinned commit SHA still checks
      out clean.
   2. `uv sync --extra cuda`, then start `uv run vetbot-web`. Confirm
      Whisper, sentiment, Piper EN + HU all warm-load successfully
      from the FastAPI lifespan logs.
   3. Empty `output/`, save its previous contents into a dated
      `output_archive/` folder so the new session writes appear in
      isolation.
   4. Open Chrome to `http://127.0.0.1:8000`, run one self-test
      utterance per language to confirm round-trip audio.
   5. Open a fresh "session NN" page in a notes file.
2. **Consent and recording**
   1. Read the consent paragraph aloud, collect signature on the
      paper log sheet (kept for the *Kiértékelési jegyzőkönyv* deliverable).
   2. Start screen capture (OBS) and a secondary phone audio recording
      as backup.
3. **Briefing**
   1. Explain in one minute: "This is a voice chatbot that books an
      appointment. Hold the mic button or `Enter`, speak, release. The
      panel on the right shows what the bot has understood so far."
   2. *Do not* explain the slot names, the phone digit-readback, or
      the two-language support — those are exactly what we are testing.
4. **Microphone test** (~30 s): the participant says one short
   sentence; the moderator confirms the transcript appears and audio
   plays back. If it does not, switch to the backup mic and retry.
5. **Task 1** (V or H, per the randomised order)
   1. Hand the printed task card. The participant reads silently for
      up to 1 minute, then begins when ready.
   2. The moderator stays silent except to repeat the *think-aloud*
      prompt at most twice: "Tell me what you're expecting to happen
      next."
   3. The observer records, per slot: (a) attempt count, (b) whether
      the bot's interpretation matched the card, (c) whether the user
      had to explicitly correct a slot, (d) any audio failure.
   4. The task ends when the WebSocket emits a `saved` event *or*
      after 12 minutes, whichever is first. If the 12-minute cap is
      hit, log it as "incomplete — timeout".
6. **SUS for Task 1** (~3 min): the participant fills the 10-item SUS
   on paper (HU translation for HU participants).
7. **Break** (~2 min): water, no chatbot talk.
8. **Task 2** (the other domain): same procedure as Task 1.
9. **SUS for Task 2**.
10. **Debrief (10 min)**: the moderator asks the three open questions
    from §1.3 and notes the answers verbatim. The participant is
    invited to "scroll back through the conversation" while answering.
11. **Reset for next session**
    1. Move the new files in `output/` into `output_archive/session_NN/`.
    2. Restart `vetbot-web` so each session begins from a cold model
       cache and a fresh dialogue history (the lifespan hook re-warms
       the models, but `BookingState` is per-WebSocket anyway).
    3. Clear the Chrome `localStorage` so any UI preferences reset.

### 2.6 Data collection

| Channel | What it captures | Tool |
| --- | --- | --- |
| Server log | per-utterance ASR, sentiment, tool calls, timing | `vetbot-web` stderr → file |
| WebSocket frames | every JSON frame to/from the browser with timestamps | small Python tap script around the existing `/ws` route |
| Output JSON | the final booking record per session | `output/appointment_<domain>_<ts>.json` |
| Screen capture | full UI + slot panel + audio | OBS, 720p |
| Audio backup | participant voice, room audio | phone recorder on the desk |
| Observer sheet | per-slot success matrix, anomalies, time markers | paper form (one per task) |
| SUS forms | 10 Likert items, one per task | paper, scanned afterwards |
| Open-question notes | verbatim answers to the 3 debrief questions | notes file |
| Consent log | signed participation record | paper, scanned for the *jegyzőkönyv* deliverable |

*"What if a participant does not consent?"* — they are thanked and the
session does not happen. Recording-without-consent is not an option.
Audio backup and screen capture are both stopped before the consent
paragraph is read, and only re-started after the form is signed.

### 2.7 Documentation

After every session the moderator spends 10 minutes writing a short
per-session memo (≤ ½ page) into `eval/session_NN.md`:

- session ID, date, participant code (no real name), language order,
  domain order;
- what worked, what failed;
- top three anomalies for the engineer to investigate next week.

The point is not to write a long report but to make sure the rest of
the team is informed before the next session starts.

### 2.8 Data analysis

After all sessions are complete:

1. **Aggregate the raw data** into a single spreadsheet: one row per
   (participant × domain) session, columns for the quantitative
   metrics from §1.3 and the SUS score.
2. **Statistical summary**:
   - Mean and median booking duration per domain and per language.
   - Slot-level accuracy histograms.
   - SUS overall mean ± standard deviation; per-question breakdown to
     find which usability dimension scored worst.
3. **Identify usability problems**: list each anomaly observed in ≥ 2
   sessions, tag it with the affected *Five Es* criterion, and rank
   by severity × frequency.
4. **Account for user characteristics**: split metrics by language
   (HU vs. EN), by self-reported voice-assistant familiarity, and by
   domain (vet vs. hairdresser) to see whether any problem is
   population-specific rather than systemic.
5. **Recommend changes** → *Evaluation report (Kiértékelési beszámoló)*:
   one page covering expectations, observations, conditions, and
   concrete UI / prompt / dialogue changes.

### 2.9 Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Whisper mis-transcribes the participant's phone number | The digit-readback + `confirm_phone` tool is exactly the recovery path under test; if it also fails, log it as a severity-1 finding. |
| Anthropic API outage during a session | The session is rescheduled to the slack slot at the end of the week; no offline fallback is offered because the system relies on Haiku for tool dispatch. |
| Audio device fails mid-session | Switch to the backup wired microphone; if that also fails, reschedule. |
| Participant becomes uncomfortable | The consent form explicitly grants the right to stop at any time; the moderator stops recording immediately on request and discards the partial data. |
| GPU memory pressure causes Whisper to fall back to CPU mid-session | Pre-flight the GPU before the participant arrives; if it happens during a session, log and continue — the CPU `small` model is still usable and the latency hit is part of what we are measuring. |
