"""All VLM / LLM prompts used by the forgery-detection pipeline.

Centralised so:
- agent_pipeline.py / tools.py stay focused on logic, not 1000-line prompt blobs;
- the paper Appendix can quote this single file as the canonical prompt set.

Grouped by stage:
    Stage A: retrieval        (EXTRACT, CLUSTER, REFLECT, RANK, VERIFY, ACTION)
    Stage B: forgery analysis (COMPARE, INFER)
    Stage C: scoring          (JUDGE)
"""

from __future__ import annotations


# ============================================================
# Stage A - retrieval agent
# ============================================================

PROMPT_REFLECT_REFINE = """You are a video retrieval strategist working inside an iterative search loop. Previous YouTube queries did NOT find a useful source video.

CONTEXT OF FAILURE:
- Previously tried queries: {prev_queries}
- WRONG candidate titles returned by those queries: {candidate_titles}

PRIOR COT ANALYSIS OF THE INPUT VIDEO:
- Physical observations: {physical_observations}
- Logical analysis: {logical_analysis}
- Search intent: {search_intent}
- Entities summary: {entities_summary}

YOUR JOB: look at the input frames again, study the failed candidate titles, diagnose why retrieval drifted, and produce ONE stronger follow-up query.

THINK STEP BY STEP (write your full reasoning in the `reasoning` field):
Step 1. Re-understand the input video from the frames:
  - What are the most diagnostic anchors: named people, program names, event names, competition names, locations, dates, logos, uniforms, scene types, or specific actions?
  - Which of those anchors are most likely to appear in the original uploader's title or in strong retrieval results?

Step 2. Diagnose the failure mode:
  - Did previous queries over-focus on the wrong person, event, program, location, or date?
  - Were they too broad, too narrow, or anchored on a side detail instead of the main verifiable fact?
  - Did the wrong candidate titles suggest a different topic cluster than what the input frames actually show?

Step 3. Produce `negative_keywords` conservatively:
  - Add ONLY specific entities, titles, locations, dates, or topic anchors that clearly pulled retrieval into the wrong result cluster.
  - Do NOT add broad genre words that could still belong to the right title.
  - When unsure, leave a term out. It is better to keep a reusable word than to block a potentially correct one.

Step 4. Generate 1 fresh search query:
  - Choose the single best angle that is MOST DIFFERENT from all previously tried queries.
  - You are free to use ANY retrieval angle: entity/event/title, location/date/program, visual cue, content category, or any combination.
  - The query must NOT repeat the same approach that already failed.

QUERY RULES:
- The query must be plain YouTube search text, 5-12 words.
- It should sound like a plausible title fragment or high-value search phrase, not a forensic note.
- Avoid reusing any term in `negative_keywords` unless it is part of an explicit exclude operator like `-term`.

OUTPUT JSON FORMAT:
{
  "reasoning": "Step 1: <what the video most likely shows>. Step 2: <why previous searches failed>. Step 3: <which specific anchors were treated as negatives>. Step 4: <how the new query improves retrieval>.",
  "reflection": "Short explanation of what retrieval signal was re-prioritized.",
  "negative_keywords": ["wrong_entity_or_topic_1", "wrong_entity_or_topic_2"],
  "new_queries": ["single best query"]
}"""


PROMPT_VERIFY_MATCH = """You are a forensic visual judge. You will see frames from two sources:
Group A: The target video we are trying to source. (First {num_target} frames)
Group B: A candidate original video from YouTube. (Remaining frames)

Your Task: Determine if Group A is derived from the SAME original source footage as Group B.

Evaluation Criteria:
1. Ignore Contextual Manipulations: Group A may have added black borders, fake subtitles, color grading, or swapped faces. IGNORE these differences.
2. Focus on Immutables: Background layout, physical text on walls/ads, specific kinetic motion, camera angle.
3. If the background, camera angle, and physical environment are identical, it is a MATCH.
4. Provide a confidence score in [0,1] reflecting your certainty.
5. If you saw the closest matching segment in Group B, point its approximate time window in seconds (relative to Group B start). If unsure, set start_sec=0 and end_sec=0.

OUTPUT JSON FORMAT:
{
  "reasoning": "Step-by-step comparison of the physical environments.",
  "is_match": true,
  "confidence": 0.0,
  "most_similar_segment": {"start_sec": 0, "end_sec": 0}
}"""


PROMPT_COT_RETRIEVAL = """You are an elite Open-Source Intelligence (OSINT) visual analyst. You will see 64 uniformly-sampled frames from a SINGLE potentially-forged video. Your job is a deep chain-of-thought analysis across ALL frames to identify what the original source video(s) were and generate YouTube search queries to find them.

CRITICAL CONTEXT: The input video is a SUSPECTED FORGERY assembled from real source clips. The forger has almost certainly ADDED their own caption / subtitle / UI layer on top of the real footage. THE ORIGINAL SOURCE VIDEO DOES NOT CONTAIN THOSE OVERLAID WORDS. Treating overlaid text as if it were part of the source is the single biggest failure mode of this system.

ANTI-SPOOFING RULES (violating these guarantees a miss):
1. The following on-screen text is OVERLAY (forger-added) by default and MUST be ignored when forming any query, unless you have very strong evidence it is physically printed in the scene:
   - bottom-of-frame subtitle bars / bilingual translation banners (especially CJK-translation pairs in black bands)
   - floating dramatic captions narrating events ("the old world collapsed", "lost control of the skies", "physical collapse is irreversible", etc. - narrative voice does not match the genre of the underlying clip)
   - video-player UI: progress bars, play / pause buttons, timestamps like "1:51 / 5:42", three-dot menus, fullscreen icons, channel watermarks added by reposters
   - news-ticker banners and "BREAKING" stripes laid over unrelated b-roll
   - any text whose font / colour / horizontal alignment is constant across UNRELATED scenes (telltale of a single overlay layer)
2. PHYSICAL text (acceptable to use) means text that is part of the real world being filmed: printed on signs, packaging, jerseys, ad boards, license plates, street signs, brand logos on devices in the scene (a brand logo on a TV bezel, a print on a shoe). Use ONLY this kind of text in queries.
3. Focus EXCLUSIVELY on the physical world: recognizable people, distinctive uniforms / clothing brands, venue type, geographic markers (architectural style, vegetation, language on real signage), sport / event type, distinctive kinetic actions, on-camera presenter style, lighting era.

MULTILINGUAL RULE: If non-Latin characters (Chinese / Japanese / Korean / Arabic / Russian / Thai / Cyrillic / etc.) appear PRINTED ON A PHYSICAL OBJECT in the scene (shop sign, packaging, road sign, jersey, building, menu, t-shirt - NOT inside an overlay subtitle bar / translation banner), ONE of the four queries MUST be in that script, copying the visible characters verbatim. If non-Latin characters appear ONLY inside overlay subtitles / translation banners added by the forger, DO NOT use them - they are not from the original source and a CJK-only query built from them will land on the wrong video.

THINK STEP BY STEP (write your full reasoning in the `reasoning` field - this is your chain-of-thought across all 64 frames):

Step 0 (overlay triage - MANDATORY FIRST STEP). Scan ALL 64 frames systematically. List every piece of on-screen text you can read. For each one, classify it as PHYSICAL (printed on a real object in the scene) or OVERLAY (added on top: subtitle bar, translation banner, video-player UI, caption added by reposter). Put the OVERLAY strings into `forbidden_overlay_text` - those strings AND any phrase derived from / paraphrasing them are FORBIDDEN in every query. If unsure, classify as OVERLAY (safe default).

Step 1 (temporal cross-frame analysis). Examine the sequence of 64 frames as a timeline:
  a) How many DISTINCT scenes / environments / camera setups appear? List them briefly.
  b) Are there abrupt scene transitions that suggest cross-video splicing (sudden changes in lighting, resolution, aspect ratio, color grading, or environment)?
  c) What is the estimated number of original source videos that were spliced together?
  d) Identify temporal manipulation: out-of-order scenes, repeated segments, selective omissions.
  This becomes `temporal_analysis`.

Step 2 (physical clue extraction per scene). For EACH distinct scene identified in Step 1, catalogue physical clues IGNORING all overlay text from Step 0:
  - Recognizable people (face, build, clothing brands printed on fabric)
  - Distinctive objects, tools, equipment, food, vehicles
  - Venue / location markers (architecture, vegetation, signage language, indoor/outdoor)
  - Activity / event type (sport, cooking, construction, interview, etc.)
  - Lighting era, camera quality, broadcast graphics that are PART of the source footage
  - Seasonal / holiday / geographic cues
  This combined catalogue becomes `physical_observations`.

Step 3 (forgery-aware reasoning). Based on the overlay text (Step 0) and the physical content (Steps 1-2):
  - What was the forger's likely intent? (narrative re-framing, emotional manipulation, context switching?)
  - What do the ORIGINAL source videos most likely contain, before the forger's overlay was added?
  - If the overlay text tells a dramatic story but the physical scene shows something mundane, the overlay is almost certainly fake.

Step 4 (title guessing). PRETEND YOU ARE THE UPLOADER of each ORIGINAL YouTube video. Imagine the forger's caption / subtitle / UI layer is completely erased - what does the bare footage show? For each estimated original source, answer:
  a) What CATEGORY of YouTube content is this? (cooking, tutorial, travel, news, challenge, vlog, compilation, sports, comedy, review, science, art, nature, etc.)
  b) What 3-5 GENERIC words would a VIEWER type into YouTube search to find this kind of content?
  Then write 2-3 LITERAL TITLE GUESSES the original uploader would have used FOR THE BARE FOOTAGE ONLY. They MUST NOT contain or paraphrase any string in `forbidden_overlay_text`.

CRITICAL: YouTube creators use SHORT, CONCEPTUAL titles - NOT forensic descriptions of specific objects. A video a forensic analyst would call "man in forest using specific-brand fire-starting tool" is titled "10 Tips for Making Fire in the Wild" by its creator. You must think like the CREATOR, not the analyst.

Step 5 (query generation - PER SOURCE). For EACH estimated source identified in Step 1, derive EXACTLY 1 SEARCH QUERY from your best title guess. This query must be the single most likely title fragment that would match the original YouTube video. Choose whichever of your title guesses from Step 4 is the most specific and concrete. NONE of the queries may contain or paraphrase any string in `forbidden_overlay_text`.

QUERY REQUIREMENT: The query MUST be a direct shortened version (5-12 words) of your BEST title guess from Step 4. Title-derived queries are the MOST LIKELY to match the real source video. Plain YouTube search text, 5-12 words, no labels, no JSON, no quotes.
If `estimated_sources` is 1, output exactly 1 source group with 1 query. If `estimated_sources` is 2, output 2 source groups with 1 query each (2 total), etc.

OUTPUT JSON FORMAT:
{
  "reasoning": "Step 0: <overlay-vs-physical triage across all 64 frames>. Step 1: <temporal analysis: scene count, transitions, splicing evidence, per-source scene description>. Step 2: <physical clues per scene>. Step 3: <forgery reasoning: what forger likely changed>. Step 4: <title guesses per source>. Step 5: <which title guess was chosen and why>.",
  "forbidden_overlay_text": ["overlay string 1", "overlay string 2", "..."],
  "physical_observations": "Concise list (3-5 sentences) of physical clues across all scenes, with overlay text excluded.",
  "temporal_analysis": "Description of scene transitions, estimated source count, any splicing/temporal manipulation evidence.",
  "estimated_sources": <1-6>,
  "source_queries": [
    {
      "source_label": "source_1: brief description of this source's scene/content",
      "queries": ["single best search query derived from title guess"]
    },
    {
      "source_label": "source_2: brief description of this source's scene/content",
      "queries": ["single best search query derived from title guess"]
    }
  ]
}"""


PROMPT_COT_RETRIEVAL_V3 = """You are a video forensic analyst. You will see 64 frames uniformly sampled from a video that may contain misinformation.

Your task is to deeply understand the video content, identify key entities and potential logical anomalies, and produce the initial retrieval plan for finding original or closely related source videos on YouTube.

Think step by step (write your full reasoning in the `reasoning` field):

== Step 1: Content Understanding ==

Examine the 64 frames as a timeline:

a) What different scenes, environments, or camera setups appear in the video? Briefly describe each one.

b) Identify the key entities and information in the video:
   - People: Who appears? Can you identify specific individuals such as celebrities, athletes, presenters, politicians, or creators?
   - Locations: Are there identifiable landmarks, venue cues, signs, architectural styles, neighborhoods, or countries?
   - Events: What is happening? A sports match, program segment, performance, interview, protest, news event, travel clip, everyday scene?
   - Time: Can you infer dates, seasons, eras, or broadcast periods?
   - On-screen text: What readable text seems semantically meaningful for identifying the content, such as names, event titles, program logos, scoreboards, venue text, headlines, or captions that appear integral to the footage? Ignore obvious platform chrome like playback controls, timestamps, or menus.

c) Are there discontinuities between scenes, such as sudden changes in environment, production style, resolution, or topic, suggesting multiple original sources?

== Step 2: Logical Reasoning ==

Combine visual information, text information, and common sense to look for potential anomalies or verification targets. Consider ALL of the following:

NARRATIVE-LEVEL ANOMALIES:
- Are the facts claimed or implied by the video internally consistent?
- Are there unreasonable links between people, locations, times, or events across scenes?
- Are there scenes that look edited together from different sources?

AI-GENERATION ANOMALIES:
- Do any objects appear visually inconsistent with their surroundings (wrong lighting, unnatural texture, mismatched brand/style for the context)?
- Are there objects or products that seem contextually out of place (e.g., a branded product appearing in a setting where it would not naturally exist)?
- Do any faces look inconsistent with the person's body, clothing, or prior appearance in other frames?
- Are there segments where visual quality, motion dynamics, or rendering style suddenly differs from the rest?

VERIFICATION TARGET:
- Is there a specific claim, identity, location, or event that is most worth verifying first?
- If there is no obvious contradiction, explain what the most valuable verification target would be.
- If AI-generation is suspected, note which elements may be synthetic so they are NOT used as retrieval anchors.

== Step 3: Retrieval Planning ==

Based on Step 1 and Step 2, determine the single best initial search query to begin retrieval:

- What is the most promising verification target to search for first, and why?
- Estimate how many distinct original sources are likely present (this informs later rounds, but you only need ONE query now).
- Produce exactly ONE high-value YouTube search query of 5-12 words targeting the most verifiable and specific element you identified.

Query strategy:
- Prioritize specific people, event names, program names, competition names, locations, dates, and distinctive scene cues.
- If a named person + event is identifiable, combine them directly.
- If a program, broadcast, or competition is identifiable, use that as a strong anchor.
- If no specific entity is reliable, use the content category plus the most distinctive visual cue.
- The query should sound like a plausible YouTube title fragment or high-value search phrase, not a forensic note.
- Do NOT generate multiple queries. Subsequent queries will be generated adaptively based on retrieval results from this first attempt.

OUTPUT JSON FORMAT:
{
  "reasoning": "Step 1: <content understanding>. Step 2: <logical reasoning>. Step 3: <why this query is the best starting point>.",
  "entities": {
    "people": ["identified people"],
    "locations": ["identified locations"],
    "events": ["identified events/programs/competitions"],
    "text_claims": ["key facts or labels conveyed by on-screen text"]
  },
  "logical_analysis": "Logical analysis of the video content, including contradictions found or the main fact that should be verified first.",
  "physical_observations": "Summarize the most searchable visual clues in 3-5 sentences.",
  "search_intent": "What this retrieval should verify or reconstruct first.",
  "estimated_sources": 1,
  "source_queries": [
    {
      "source_label": "source_1: brief description of the most promising verification target",
      "queries": ["single best first query (5-12 words)"]
    }
  ]
}"""


PROMPT_COARSE_RELEVANCE = """You are a coarse filter in a video retrieval pipeline.

You will see TWO frame groups:
- Group A: frames from the SUSPECTED FORGERY video.
- Group B: frames from a CANDIDATE YouTube video.

CONTEXT FROM PRIOR COT ANALYSIS:
- Physical observations: {physical_observations}
- Logical analysis: {logical_analysis}
- Search intent: {search_intent}

YOUR TASK: judge whether Group B is useful retrieval evidence for Group A. The candidate can be:
- the same original footage,
- a closely related upload of the same event or scene family,
- or a video that would materially help verify the main fact or claim in Group A.

CRITICAL EVALUATION RULES:
1. This is a COARSE FILTER with HIGH RECALL. Prefer false positives over false negatives.
2. Do NOT rely on pixel-level similarity.
3. Ignore differences caused by compression, bitrate, resolution, color grading, subtitles, black bars, crop, mirror, small timing offset, and camera/viewpoint variation.
4. Focus on semantic anchors: event type, named entities, venue/program cues, recurring objects, role relations, action pattern, and storyline.
5. If Group B could plausibly help verify the same identity, event, location, or claim as Group A, mark relevant.
6. Only mark irrelevant when topics/scenes are clearly different.

OUTPUT JSON FORMAT:
{{
  "reasoning": "Brief semantic comparison between Group A and Group B.",
  "is_relevant": true
}}"""


PROMPT_FINE_FORGERY_POINTS = """You are a forensic video analyst comparing:
- Group A: frames from a SUSPECTED FORGERY video
- Group B: frames from a CANDIDATE source video

PRIOR CONTEXT:
- Physical observations: {physical_observations}
- Logical analysis: {logical_analysis}
- Search intent: {search_intent}
- Entities summary: {entities_summary}

YOUR TASK: Perform a detailed cross-group comparison and extract forgery points that describe WHAT IS FALSE in Group A, using Group B as ground-truth evidence.

Forgeries fall into three broad categories — consider ALL of them:

A) AI GENERATION: faces replaced by deepfake (identity swap), objects added/removed/replaced by generative editing, or contiguous AI-generated segments inserted into real footage. Clues: identity mismatch (same body/background but different face), objects that exist in A but not in B (or vice versa), segments in A with no corresponding content in B that show telltale synthetic continuity.

B) SINGLE-SOURCE EDITING: the same source footage is editorially altered — scenes selectively omitted to change meaning, temporal order reversed to flip causality, or shots rearranged to create a misleading emotional impression.

C) MULTI-SOURCE SPLICING: footage from Group B is combined with footage from other unrelated sources to fabricate a narrative, distort the severity of an event, or place real footage into a false context (wrong time, location, or attribution).

THINK IN THIS ORDER (write reasoning in source_description):
1. IDENTITY & OBJECT ALIGNMENT: Compare the people, faces, and key objects between Group A and Group B. Are the same individuals present? Are faces consistent? Are objects added, removed, or replaced?
2. TEMPORAL & STRUCTURAL ALIGNMENT: Do the scenes in Group A appear in the same order and completeness as in Group B? Are segments missing, reordered, or inserted?
3. CONTEXTUAL ALIGNMENT: Does Group A preserve the original context (time, location, event, attribution) shown in Group B, or does it reframe the footage into a different narrative?
4. SYNTHESIS: What is the overall false impression that Group A creates, and which category (AI generation / single-source editing / multi-source splicing) best describes the manipulation?
5. DEDUPLICATE: each point must cover a unique manipulation aspect.

OUTPUT JSON FORMAT (strict, no extra text, NO code fences):
{{
  "source_description": "1-2 sentences: semantic relation between Group A and Group B plus the primary forgery mechanism identified.",
  "points": [
    {{
      "description": "Forgery point: what Group B shows as ground truth, what differs in Group A, and the false impression this creates."
    }}
  ]
}}

CONSTRAINTS:
1. `points` should contain 3 to 5 entries when evidence supports it.
2. No duplicate points with paraphrased wording.
3. Ignore pure quality/viewpoint/compression differences unless they support a manipulation claim.
4. At least one point should provide a high-level synthesis of the overall manipulation (not just one local edit or one replaced element).
5. For identity swaps: describe the original person's appearance in Group B (hair color/style, clothing, build, accessories) and state that Group A replaces their face with a different person's face via AI. Do this for EACH person whose face is swapped.
6. For object manipulation: describe the original object in Group B (what it is, where it appears, its role in the scene) and state what it has been replaced with or altered into in Group A. Do this for EACH manipulated object.
7. Output VALID JSON only. No markdown fences, no commentary."""


PROMPT_DEEPSEARCH_NEXT_STEP = """You are the decision-maker in a deep-search loop for video forgery analysis. You have been searching YouTube for original source videos and collecting forgery evidence.

CONTEXT (do NOT echo back):

- Round status:
{round_status}

— What the video looks like (from COT analysis of the input video):
- Physical observations: {physical_observations}
- Logical analysis: {logical_analysis}
- Search intent: {search_intent}
- Entities summary: {entities_summary}

— Estimated source families / scene groups:
{source_descriptions}

— Forgery evidence collected so far:
{collected_points}

— Videos examined so far:
{examined_videos}

— Previous search queries tried:
{prev_queries}

YOUR JOB (two tasks in one response):

TASK 1 — SUFFICIENCY JUDGMENT: Compare what you KNOW about the input video against what you've COLLECTED. Judge whether the evidence is SUFFICIENT.

THINK STEP BY STEP for sufficiency:
1. Based on the source descriptions, how many distinct source families or scenes likely matter? Have we found evidence for each?
2. Based on physical_observations and logical_analysis, what identities, events, locations, or claims must be verified? Do the collected points cover them?
3. Does the examined evidence explain the main misleading interpretation or unresolved factual question?
4. Is there any major scene, claim, or verification target that NO collected point covers?

CRITICAL RULES:
- For a SINGLE dominant source: do NOT stop merely because you found a topically related video. The examined evidence should strongly suggest the same program, same event, same source family, or a near-duplicate upload of the same underlying footage.
- If the current evidence mainly highlights that Group A and Group B are from different eras, different shows, different productions, or different contexts, that is evidence of topical relatedness, NOT sufficient source resolution. Keep searching.
- For MULTI-SOURCE or MULTI-SCENE cases: you need evidence that covers EACH major source family or unresolved scene cluster.
- Prefer precision over early stopping. It is acceptable to continue searching when the current best evidence is only "related but not same-source enough."
- If the round status says this is the FINAL round, you MUST set `is_sufficient=true`, leave `missing_description` and `next_keyword` empty, and do not request another search.

TASK 2 — NEXT KEYWORD (only if NOT sufficient): Generate the SINGLE best YouTube search query to find the next missing original source.

THINK STEP BY STEP for keyword generation:
1. Which source family, identity, event, location, or claim is still unresolved?
2. What specific distinguishing cue would most likely retrieve useful evidence for it?
3. Generate ONE search query (5-12 words) using the strongest identifying anchor available.
4. Prefer a query that could plausibly match a real YouTube title or high-value search phrase.

OUTPUT JSON FORMAT:
{{
  "reasoning": "Step-by-step: what the forgery contains vs what evidence is collected, whether the current evidence is same-source enough, and what's still missing.",
  "is_sufficient": true | false,
  "missing_description": "If insufficient, describe exactly what evidence is missing. If sufficient, empty string.",
  "next_keyword": "If insufficient, ONE best search query (5-12 words). If sufficient, empty string."
}}"""


# ============================================================
# Stage B - forgery analysis (NEW)
# ============================================================

# The judge compares 3-5 GT points against 3-5 predicted points and scores
# whether each GT point is matched by a prediction, primarily on the
# misleading-point dimension.

# ============================================================
# Stage C - LLM-as-judge (NEW)
# ============================================================

PROMPT_JUDGE_POINTS = """You are a strict but fair evaluator for video forgery analysis. You will be given a ground-truth (GT) list of 3-5 forgery points and a predicted list of 3-5 forgery points. For each GT point, decide whether any prediction matches it.

Important framing:
- GT points and predictions may be written at different abstraction levels.
- Some items are narrative-level accusations: the edit constructs a false moral, political, social, or causal interpretation.
- Some items are event-level findings: the edit splices, reorders, re-contextualizes, or rebinds specific scenes, audio, or actions.
- A good prediction may express the same core misleading point either directly at the narrative level or indirectly through concrete event evidence.

Scoring rules (important; follow strictly):
1. Each GT point can be matched at most once. Use a one-to-one best assignment.
2. Evaluate every GT/pred pair on two dimensions:
   - Dimension A: manipulation method. Examples: splicing unrelated clips, reordering scenes, removing narration, binding unrelated audio, selective omission, context switching.
   - Dimension B: misleading point / false reframing. Examples: portraying someone as rule-breaking, constructing a negative personality arc, implying opportunistic recruitment, or creating a moral-judgment framing.
3. Dimension B is primary, but judge it flexibly across abstraction levels:
   - Direct narrative match: If the prediction states substantially the same false reframing or accusation as the GT, mark `matched_dim_misleading=true` and `verdict=1`.
   - Event-to-narrative support: If the GT is narrative-level, and the prediction identifies the concrete edited event(s) that materially support that same narrative reframing, you may still mark `matched_dim_misleading=true` and `verdict=1` even if the prediction is more concrete and less interpretive.
4. "Roughly consistent" allows paraphrase, synonymy, wording differences, partial compression of multiple scenes into one point, and different abstraction levels, as long as the core misleading implication is aligned.
5. Be conservative about over-crediting:
   - Do not give credit just because both items mention the same event.
   - Do not give credit just because both items describe editing manipulation in general.
   - Do give credit when the prediction captures the same accusation, reframing, or a concrete event-level mechanism that clearly underpins that accusation.
6. Report `matched_dim_method` truthfully even when `verdict=1` is driven mainly by Dimension B.
7. Every GT point should receive the best available predicted match. If multiple GT points compete for the same prediction, choose the assignment that gives the best overall scoring.
8. Output strict JSON only. Do not use markdown fences.

GT (3-5 points):
{gt_block}

PREDICTION (3-5 points):
{pred_block}

Output JSON:
{{
  "hits": <integer from 0 to number_of_gt_points>,
  "score": <hits / number_of_gt_points as a float>,
  "matches": [
    {{
      "gt_idx": <0-based GT index>,
      "pred_idx": <0-based prediction index or null if no match>,
      "verdict": <0|1>,
      "matched_dim_method": <true|false>,
      "matched_dim_misleading": <true|false>,
      "reason": "One concise English sentence explaining whether the method matches, whether the misleading point matches, and the key mismatch if any."
    }}
  ],
  "comment": "1-2 concise English sentences summarizing the biggest strength and biggest weakness of the predictions."
}}"""
