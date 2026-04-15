"""
Define your evaluation test cases here.
Each case needs a transcript (string) and a query (string).
No ground-truth answers needed — RAGAS evaluates faithfulness and answer relevancy
purely from the retrieved context and LLM answer.
"""

TEST_CASES = [
    # --- Factual intent ---
    {
        "transcript": """\
[00:00:10] Alice: Good morning everyone. Let's start with the Q3 budget review.
[00:00:20] Bob: Sure. We overspent on cloud infrastructure by 15% this quarter.
[00:00:35] Alice: That's significant. What drove the spike?
[00:00:45] Bob: Mostly the new data pipeline we rolled out in July. It's using more compute than estimated.
[00:01:00] Carol: We flagged that risk back in June but the timeline couldn't slip.
[00:01:15] Alice: Understood. Bob, can you model a 10% reduction scenario for next quarter?
[00:01:25] Bob: Yes, I'll have that ready by Friday.""",
        "query": "What caused the cloud infrastructure overspend?",
    },

    # --- Speaker intent ---
    {
        "transcript": """\
[00:00:05] Alice: Let's go over the roadmap priorities.
[00:00:15] Bob: I think we should push the mobile feature to Q2.
[00:00:25] Alice: Agreed. The backend stability work needs to come first.
[00:00:40] Carol: I can take ownership of the API refactor if that helps.
[00:00:55] Bob: That would be great Carol. I'll focus on performance testing then.
[00:01:10] Alice: Perfect. Let's lock this in.""",
        "query": "What did Bob say about the roadmap?",
    },

    # --- Synthesis intent ---
    {
        "transcript": """\
[00:00:10] Alice: Alright let's wrap up. What are our action items?
[00:00:20] Bob: I'll send the revised budget model to Alice by Friday.
[00:00:30] Carol: I need to follow up with the vendor on the SLA breach.
[00:00:45] Alice: And I'll schedule the retrospective for next Tuesday.
[00:01:00] Bob: Also, someone needs to update the project tracker.
[00:01:10] Carol: I can do that today.""",
        "query": "What are the action items from this meeting?",
    },

    # --- Temporal intent ---
    {
        "transcript": """\
[00:00:05] Alice: Let's start the standup.
[00:00:15] Bob: I finished the login module yesterday.
[00:00:25] Carol: I'm still blocked on the database migration.
[00:00:40] Alice: We need to unblock Carol. Bob can you help?
[00:00:50] Bob: Sure, I'll pair with Carol after this call.
[00:05:00] Alice: Moving on, let's talk about the release.
[00:05:15] Bob: Release is on track for Thursday.
[00:05:30] Carol: I'll need the migration done before I can sign off on the release.""",
        "query": "What was discussed in the last 3 minutes of the meeting?",
    },
]
