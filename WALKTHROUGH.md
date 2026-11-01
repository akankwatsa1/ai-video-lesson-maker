# AI Lesson Generator Studio (v6.6) — Complete Feature & Technical Walkthrough

> **Purpose:** This document is the definitive, comprehensive guide to the **AI Lesson Generator Studio (v6.6)**. Any AI agent or developer reading this walkthrough will have an exact, complete understanding of every feature, how each feature is structured, how the frontend and backend communicate, how the cloud GPU worker renders assets, and how data flows through the entire system.

---

## Table of Contents
1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Frontend Member Authentication & CRM System](#2-frontend-member-authentication--crm-system)
3. [Step 1: Lesson Configuration & Content Ingestion (5 Creation Methods)](#3-step-1-lesson-configuration--content-ingestion-5-creation-methods)
4. [Step 2: Interactive Script Review & Timeline Editor](#4-step-2-interactive-script-review--timeline-editor)
5. [Step 3: Real-Time Progress, In-Browser Preview & Permanent Downloads](#5-step-3-real-time-progress-in-browser-preview--permanent-downloads)
6. [Automated Hands-Free YouTube SEO Kit & Cloud Publisher](#6-automated-hands-free-youtube-seo-kit--cloud-publisher)
7. [User & Admin History Management (Single & Batch Deletion, Resumption)](#7-user--admin-history-management-single--batch-deletion-resumption)
8. [WordPress Admin Control Center (6 Dedicated Tabs)](#8-wordpress-admin-control-center-6-dedicated-tabs)
9. [RunPod GPU Worker Engine (`handler.py` & `lipsync_engine.py`)](#9-runpod-gpu-worker-engine-handlerpy--lipsync_enginepy)
10. [Critical System Rules & Handover Guidelines](#10-critical-system-rules--handover-guidelines)

---

## 1. System Overview & Architecture

The system is a production-grade educational studio that converts text, documents, URLs, or recorded voice lectures into broadcast-quality educational videos with an AI presenter, animated PowerPoint slides, data charts, HD B-roll footage, subtitles, background music, audio mastering, custom thumbnails, and direct YouTube cloud publishing.

```
+-----------------------------------------------------------------------------------+
|                           1. WORDPRESS PLUGIN FRONTEND                            |
|  Shortcode: [ai_lesson_generator]                                                 |
|  - Member Auth (Email, WP 1-Click, Google OAuth)                                 |
|  - Step 1: Config, 5 Script Ingestion Modes, Voices, Avatars, Templates           |
|  - Step 2: Interactive Timeline Editor with Block Controls & Dynamic CTA         |
|  - Step 3: Live Progress, In-Browser HTML5 Player, Auto-SEO, 1-Click YouTube      |
|  - User History: Permanent Downloads, Batch Deletion, Job Resumption              |
+-----------------------------------------------------------------------------------+
                                         |
                                         | (AJAX + Nonce / Session Token)
                                         v
+-----------------------------------------------------------------------------------+
|                           2. WORDPRESS BACKEND CONTROLLER                         |
|  File: includes/class-api-handler.php (AILG_API_Handler)                          |
|  - LLM Cascade: DeepSeek Primary -> Google Gemini Auto-Cascade -> Fallback Engine |
|  - Content Extraction: PDF/DOCX/PPTX non-backtracking scanner, Jina URL fetcher   |
|  - Media Presigning: AWS S3 SigV4 for Cloudflare R2                               |
|  - Job Dispatch: RunPod Serverless REST API with 60-min executionTimeoutMs        |
|  - YouTube Data API v3: OAuth 2.0 Resumable Cloud Uploader + Thumbnail + .SRT    |
+-----------------------------------------------------------------------------------+
                         /                               \
                        / (Upload Assets / Fetch Media)   \ (Dispatch Job)
                       v                                   v
+------------------------------------+    +-----------------------------------------+
|      3. CLOUDFLARE R2 & AI         |    |         4. RUNPOD SERVERLESS GPU        |
|  Bucket: video-making              |    |  Container: CUDA + PyTorch + FFmpeg     |
|  - /uploads (Avatars, Voices)      |    |  Files: handler.py, lipsync_engine.py   |
|  - /library (Media presets)        |    |  - R2 Block Checkpointing Engine        |
|  - /outputs (MP4, JPG, SRT)        |    |  - Head-Safe Avatar Framing             |
|  - /progress/{job_id}.json         |    |  - Video-Loop Lip-Sync Unification      |
|  - Workers AI (@cf/openai/whisper) |    |  - 4 Animated Slide Styles (Typewriter) |
|  - Flux-1-Schnell AI Visuals       |    |  - -14 LUFS Broadcast Audio Mastering   |
+------------------------------------+    +-----------------------------------------+
```

---

## 2. Frontend Member Authentication & CRM System

To isolate student/creator accounts from administrative WordPress users, the plugin provides a dedicated member management subsystem:

1. **Dedicated Database Table (`wp_ailg_users`):**
   - Created automatically via `AILG_API_Handler::ensure_users_table()`.
   - Stores `full_name`, `email`, `phone`, `organization`, `password_hash`, `auth_provider` (`email` vs `google`), `google_sub`, `session_token`, `jobs_count`, `last_login`, and `created_at`.
2. **Triple Sign-In Pathways:**
   - **Email & Password:** Registration and login with 30-day "Remember Me" persistent cookie (`ailg_user_token`) and `localStorage` session backup.
   - **1-Click WordPress Session Quick Login:** If a user is already signed into WordPress, a prominent button allows instant sign-in without typing credentials.
   - **Dual-Mode Google Sign-In (OAuth 2.0):**
     - Uses standard Google Web Application Client ID (`.apps.googleusercontent.com`).
     - Features **Live Preflight Server Verification** (`ailg_google_login_preflight`): tests Google connectivity before redirecting, preventing `401 client_not_found` errors.
     - Supports both full-page OAuth redirect and instant Google Token Client popup fallback.
3. **Self-Service Password Reset:**
   - Users who forget their password verify their account by confirming their registered Phone Number, Full Name, or Organization, immediately setting a new password without third-party email bottlenecks.
4. **Top Bar & Session State:**
   - When signed in, displays member name, organization/email badge, total generated lessons count, and buttons for **My Previous Generations** and **Sign Out**.

---

## 3. Step 1: Lesson Configuration & Content Ingestion (5 Creation Methods)

Step 1 enables creators to customize pedagogical and visual settings across 5 distinct content sources:

### A. Media Output Format
- **Full Video Lesson (`.mp4`):** Includes presenter avatar, animated PowerPoint slides, data charts, HD B-roll cutaways, burned-in karaoke subtitles, studio backgrounds, and background music.
- **Audio-Only Lesson (`.mp3`):** High-speed podcast/voiceover mode bypassing video compilation.

### B. Five Script & Content Creation Methods
1. **AI Generate from Topic:**
   - User inputs **Lesson Title/Topic** and optional **Key Learning Objectives**.
   - Selects **Target Lesson Duration**: 1 min (~150 words), 3 min (~450 words), 5 min (~750 words), 8 min (~1200 words), or 10 min (~1500 words).
   - Driven by `draft_timeline()`: queries DeepSeek Chat with Google Gemini cascade fallback (`gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.0-flash-lite`). If all APIs time out, deterministic `build_smart_fallback_timeline()` builds a curriculum instantly.
2. **Convert Audio Lesson to Video:**
   - User uploads a pre-recorded lecture (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`).
   - Sliced/transcribed via Cloudflare Workers AI Whisper with fast 15-second cap.
   - Keeps the creator's exact recorded audio as the master audio track and layers presenter avatars, slides, charts, B-roll, and subtitles over it.
3. **Generate from Webpage URL:**
   - User pastes an article or blog post URL.
   - Handled by `extract_webpage_content()`: uses direct non-backtracking HTML scraper with automated Jina Reader (`https://r.jina.ai/`) fallback for JavaScript-rendered or protected pages.
4. **Upload Document (`.pdf`, `.docx`, `.pptx`, `.txt`, `.md`):**
   - User uploads lecture slides, syllabi, or handouts.
   - Handled by `extract_uploaded_document_text()`: fast non-backtracking PDF stream parser (skipping binary font/image streams) and native ZipArchive XML parser for DOCX/PPTX.
5. **Paste Custom Script:**
   - User pastes an entire lecture transcript. Automatically parsed into sequential ~30-second speaking blocks with matching slide bullet points.

### C. Subject Spheres & Directional Presets
Auto-configures cinematography, voice pacing, slide styles, and background soundtracks:
- **BBC / NatGeo Documentary:** Sets deep measured pacing (0.9x speed), cinematic pauses, 72% full-screen motion B-roll, and orchestral soundtrack.
- **Agriculture & Livestock:** Tailors visual prompts for farming, animal feeds, and acoustic African soundtracks.
- **Geography & Earth Science:** Prioritizes landscape B-roll and ambient soundtracks.
- **STEM & Chemistry:** Triggers SVG data charts, molecular prompts, and ambient science synths.
- **Religion & Theological Studies:** Formats reflective pacing and gospel/spiritual pads.
- **Business & Economics:** Formats executive summaries, financial bar charts, and corporate tech audio.

### D. Voice Engines, Cloning & Acoustic Modulation
1. **Studio Voice Cloning (OmniVoice):** Clones reference voice with anti-bleed conditioning in English, Kiswahili, or Bilingual.
2. **Studio HD Voice Cloning (MOSS TTS):** 48kHz high-fidelity vocal nuance for broadcast quality.
3. **Studio Neural Voices (Microsoft Edge TTS):** Pre-built zero-shot neural voices across 17 US voices, 5 UK voices, 8 African voices (Nigeria, Kenya, South Africa, Tanzania), and 4 Swahili voices (`sw-KE-RafikiNeural`, `sw-KE-ZuriNeural`, `sw-TZ-DaudiNeural`, `sw-TZ-RehemaNeural`).
4. **Vocal Acoustic Modulation:**
   - **Delivery Style:** Normal Clear Broadcast, Documentary Narrator, Warm Storyteller, Energetic Keynote, or Whisper Mode.
   - **Speed / Pacing:** 0.8x, 0.9x, 1.0x, 1.15x, 1.3x.
   - **Voice Pitch / Tone:** -15%, -8%, 0%, +8%, +15%.
   - **Teacher Voice Volume Slider:** 50% to 250% projection, normalized to YouTube standard `-14 LUFS`.

### E. Presenter Avatars, Head-Safe Framing & Video Loop Unification
- **Static Photos (`.jpg`, `.png`):** Automatically rendered with chest-up framing, multi-zone neck/shoulder breathing articulation, and 2-camera punch-in cuts.
- **Natural Presenter Video Loops (`.mp4`, `.webm`, `.mov`):**
  - Users can upload a 3–10s video loop of a presenter breathing/nodding.
  - **Single Lip-Sync Loop Mode:** `use_wav2lip_on_video_avatar` is set to `False` on video loops to eliminate duplicate or detached floating mouth overlays.
  - **Head-Safe Framing:** `crop_head_safe_region()` detects the face using MediaPipe/Haar cascades and guarantees at least 22% headroom so the presenter's forehead/hair is never cropped off.
  - **Zero Text Overlap:** Full-screen slide cards are suppressed when the presenter avatar is on screen; lower-third banners are confined to the non-avatar half of the screen.

### F. Studio Visual Templates & Blueprints
- **6 Visual Templates:**
  1. Classic Broadcast Newsroom (Navy & Cyan Glow — Left Presenter)
  2. TED Spotlight Keynote Stage (Crimson & Gold — Right Presenter)
  3. BBC / NatGeo Documentary Widescreen (Emerald & Amber — Left Presenter)
  4. Academic Chalkboard (Forest Green & Gold — Right Presenter)
  5. Daylight Minimalist Executive Studio (Pearl & Royal Blue — Left Presenter)
  6. Futuristic STEM & Tech Innovation Lab (Obsidian & Neon Violet — Right Presenter)
  7. *Auto-Rotate:* Cycles all 6 studio styles across blocks to maximize engagement.
- **3 Visual Blueprints:** Dynamic Multi-Camera, Documentary Cinema (72% B-roll), Extended Teacher Presence.

### G. Brand Logo Watermark, Captions & Soundtracks
- **Brand Watermark:** Upload custom PNG/JPG, select from presets, or paste a URL from the WordPress Site Media Library (automatically mirrored to Cloudflare R2). Selectable in 4 corners (Top-Right default).
- **Subtitles:** Active burned-in karaoke subtitles, smart broadcast mode, or clean slides only (+ standalone `.srt` closed caption file).
- **Background Music:** 10 genres (Lofi, East African Acoustic, Corporate Tech, Cinematic Inspiring, Classical Piano, Gospel Worship, Ambient Science, Upbeat Afrobeat, Folk, Jazz Lounge) with volume slider (0% to 100%) and subtle SFX transition whooshes/chimes.

---

## 4. Step 2: Interactive Script Review & Timeline Editor

Once Step 1 drafts the lesson, the creator reviews and refines every scene before rendering:

1. **Card-Based Speaking Blocks:**
   - **Spoken Script & Karaoke Captions:** Editable script text for each ~30-second block.
   - **Pause Insertion Buttons:** One-click buttons insert `[pause 1s]`, `[pause 2s]`, and `[pause 3s]` tags for natural pacing.
   - **Delivery Style per Block:** Override global delivery (e.g. deliver block 1 as energetic, block 2 as documentary, block 3 as slow).
   - **Scene Layout Flag:** Toggle between *AI Lip-Sync Teacher + Studio Card*, *PowerPoint Slide Deck + AI Images*, or *Full-Screen B-Roll / Documentary Motion*.
   - **PowerPoint Slide Heading & Bullet Points:** Editable slide notes rendered dynamically on screen.
   - **B-Roll Search Keywords:** Custom search query for Pexels HD video clips.
   - **AI Visual Illustration Prompt:** Custom prompt sent to Flux-1-Schnell when stock footage is unavailable.
   - **Interactive SVG Data Charts:** Supports embedded chart data (bar charts, line graphs) rendered natively onto slides.
2. **Block Management:** Creators can add new speaking blocks or remove blocks with real-time re-indexing.
3. **Automated Dynamic Spoken CTA:**
   - `ensure_dynamic_cta_on_timeline()` inspects the final speaking block.
   - If a call-to-action is missing, it dynamically synthesizes a unique spoken CTA referencing the specific lesson topic, channel name, and presenter (e.g. *"If this lesson on Fractional Distillation gave you a clearer understanding, hit the Like button, share this video with a classmate, drop your questions in the comments below, and subscribe..."*).

---

## 5. Step 3: Real-Time Progress, In-Browser Preview & Permanent Downloads

When the user clicks **Confirm & Start Cloud Rendering**, the verified timeline and media are dispatched to RunPod:

1. **Lightweight Dispatch Optimization:**
   - Step 1 drafting only transmits text inputs. Step 2->3 transmits media files, avoiding upload bottlenecks.
2. **Dual-Channel Live Monitoring:**
   - Polls RunPod Serverless API status (`check_runpod_status`) every 3.5 seconds.
   - Simultaneously checks the Cloudflare R2 live progress sidechannel (`progress/{job_id}.json`).
   - Displays animated progress percentage, live step descriptions (e.g. *"Rendering Block #3 (PowerPoint Slides & B-Roll)..."*), and real-time countdown timer.
3. **In-Browser HTML5 Preview Player:**
   - Video (`.mp4`) and audio (`.mp3`) can be streamed and reviewed immediately within the browser without downloading.
4. **Permanent Cloud Download Proxy (`ailg_download_media`):**
   - Traditional presigned S3 URLs expire after 1 hour. This plugin routes downloads through:
     `admin-ajax.php?action=ailg_download_media&job_id={id}&type=video|thumbnail|srt`
   - Dynamically signs a fresh 7-day presigned R2 URL on every click, ensuring download links **never expire** as long as the record exists in history.
   - `clear_storage()` protects all active history media from 24-hour cleanup routines.

---

## 6. Automated Hands-Free YouTube SEO Kit & Cloud Publisher

The studio provides direct cloud-to-YouTube publishing with zero local downloads required:

```
[Cloudflare R2 Master Video] ───────► [WordPress Backend Streamer] ───────► [YouTube Data API v3]
[Custom 1280x720 Thumbnail]  ───────► (Zero user bandwidth consumed) ────► [Custom Thumbnail API]
[Synced .SRT Subtitles]      ───────►                                ────► [Captions API (en)]
```

1. **Automated YouTube SEO Package Generation (`generate_youtube_seo_package`):**
   - Automatically generates:
     - **SEO Title:** Topic + descriptive masterclass hook (<95 chars).
     - **Full Description:** 350+ words containing Lesson Overview, Key Takeaways, **Exact `00:00` Clickable Chapter Timestamps**, Module Summaries, Discussion Prompt, Channel Subscription CTA, and `#Hashtags`.
     - **SEO Tags:** 20 comma-separated keywords extracted from the lesson script and curriculum.
2. **Direct Cloud-to-YouTube Streaming (`publish_to_youtube`):**
   - Exchanges stored refresh tokens for fresh Google access tokens.
   - Fetches the rendered video directly from Cloudflare R2 into server memory/temp stream and performs a resumable HTTP PUT upload to YouTube Data API v3.
   - Automatically uploads the custom 1280x720 JPG thumbnail.
   - Automatically creates and attaches the `.srt` closed caption track (`English - Synced Lesson Captions`).
3. **Flexible Privacy & Scheduling:**
   - Supports **Public** (live immediately), **Unlisted** (review mode), **Private**, or **Scheduled**.
   - Scheduled publishing converts user local datetime into ISO 8601 UTC (`publishAt`) and marks the video private until release.
   - Displays direct **Watch on YouTube** and **Open in YouTube Studio** links.

---

## 7. User & Admin History Management (Single & Batch Deletion, Resumption)

Both the frontend Member Dashboard and the WordPress Admin Control Center feature a robust history manager:

1. **Job History Records (`ailg_generation_history`):**
   - Tracks `job_id`, `title`, `topic`, `status`, `progress`, `step`, `filename`, `thumbnail_filename`, `srt_filename`, `youtube_seo`, `youtube_publish`, and user attribution.
2. **1-Click Job Resumption & Retry (`regenerate_job`):**
   - If a generation stalls or is interrupted by network drops, the creator clicks **Resume / Retry Job**.
   - Dispatches the saved timeline back to RunPod.
   - **R2 Block Checkpointing:** The worker checks for existing `checkpoints/{timeline_hash}/block_XXXX_v66.mp4` files and immediately reuses already-rendered blocks, allowing interrupted jobs to finish in seconds instead of restarting from scratch.
3. **Single & Batch History Deletion (`ailg_delete_history`):**
   - **Single Delete:** Delete individual video records with confirmation.
   - **Batch Selected Delete:** Checkboxes allow bulk selection and deletion.
   - **Delete Failed / Interrupted:** 1-click cleanup of all errored or timed-out jobs.
   - **Clear All:** Completely resets generation history.
   - Best-effort R2 cleanup automatically deletes the corresponding video, thumbnail, subtitle, and progress files on Cloudflare R2.

---

## 8. WordPress Admin Control Center (6 Dedicated Tabs)

Located under **WP Admin -> AI Lesson Studio**:

| Tab | Name | Key Functionalities |
| :--- | :--- | :--- |
| **Tab 1** | **Cloud & AI Engines (`api`)** | Cloudflare R2 credentials & bucket, Workers AI API token, RunPod API Key & Endpoint ID, DeepSeek API Key, Google Gemini API Key & Model Selector (`auto`, `2.5-flash`, `2.5-pro`, `2.0-flash`, `2.0-flash-lite`), Pexels API Key, R2 24-hour cleanup button, diagnostic log download/clear. |
| **Tab 2** | **Google Sign-In & YouTube (`google_youtube`)** | Google OAuth 2.0 Client ID & Secret configuration, Live Server Verification diagnostic box, `client_secret.json` file auto-importer, Authorized JavaScript Origin & Redirect URIs display, Connect YouTube Channel OAuth button, and Default YouTube Privacy setting. |
| **Tab 3** | **Studio Branding & Logo (`branding`)** | Default Brand Logo URL (with direct WordPress Media Library integration & automatic R2 mirroring), Default Teacher/Presenter Name, Default Channel/Academy Name. |
| **Tab 4** | **Media Library (`library`)** | Upload and manage reusable presets for **Reference Voices** (with audio preview player), **Teacher Avatars** (photos and natural video loops with video preview player), and **Brand Logos** (file upload or site URL). |
| **Tab 5** | **Generation History & Scheduler (`history`)** | Comprehensive history table showing all member/admin jobs, live progress monitoring, single/batch deletion, re-download buttons, 1-click retry, and YouTube publishing/scheduling. |
| **Tab 6** | **Registered Members & CRM (`users`)** | Member accounts table from `wp_ailg_users`, search by name/email/phone/org, sortable columns, user statistics (Google vs Email accounts, total member lessons), password reset, and **Export Members to CSV**. |

---

## 9. RunPod GPU Worker Engine (`handler.py` & `lipsync_engine.py`)

The cloud worker runs inside a Docker container equipped with CUDA 12.1, PyTorch, MediaPipe, OpenCV, and FFmpeg:

1. **R2 Block Checkpoint Cache (`checkpoints/{timeline_hash}/`):**
   - Generates an MD5 hash of the timeline data, speaker voice, and visual settings.
   - Before rendering each block, checks if `block_{idx:04d}_v66.mp4` exists in R2 storage. If present, it downloads the rendered block instantly, bypassing duplicate neural inference.
2. **Audio Pipeline & Loudness Normalization:**
   - Synthesizes per-block audio using OmniVoice, MOSS TTS, or Edge-TTS (or slices uploaded audio).
   - Measures integrated loudness via FFmpeg `ebur128` filter and normalizes to `-14.0 LUFS` with `-1.5 dBFS` true peak limit.
3. **Visual Composition Engine (`lipsync_engine.py`):**
   - **Head-Safe Avatar Cropping:** MediaPipe Face Mesh detects facial landmarks and head boundaries. Dynamically enforces an upper margin ($\ge 22\%$) so presenter hair/forehead is never cropped out.
   - **Single Unified Video Loop:** Videos avoid Wav2Lip face blending to eliminate unnatural mouth artifacts or duplicate floating lips.
   - **4 Animated PowerPoint Slide Styles:**
     - *Style 0 (Typewriter Reveal):* Reveals characters progressively with an active blinking terminal cursor.
     - *Style 1 (Smooth Cascade Slide-In):* Translates bullet points smoothly from the left margin with subtle damping.
     - *Style 2 (Glassmorphic Card Pop):* Renders bullet points inside frosted glass cards with soft drop shadows and scale transitions.
     - *Style 3 (Highlight Bar Sweep):* Animates a gradient colored accent bar that sweeps across key phrases.
   - **Clean Studio Canvas:** Eliminates all artificial noise/grain filters, rendering smooth multi-point directional gradients.
4. **Thumbnail Generation (`generate_youtube_thumbnail`):**
   - Generates a 1280x720 JPG thumbnail choosing from 6 high-CTR visual layouts: Split Studio Presenter, Spotlight Keynote, Cinematic Widescreen, Chalkboard Lecture, Daylight Minimal, or Cyber STEM Lab.
   - Overlays high-contrast text, brand watermark, and presenter badge.
5. **Closed Caption Generation:**
   - Synthesizes an accurate, word-synced `.srt` file mapped to block timecodes.

---

## 10. Critical System Rules & Handover Guidelines

Any incoming AI agent or developer maintaining this codebase must adhere to the following non-negotiable rules:

1. **Zero Trade Secrets / Vendor Messaging on Frontend:**
   - Never display underlying third-party vendor names (`DeepSeek`, `Google Gemini`, `RunPod`, `Cloudflare R2`, `Cloudflare Workers AI`, `Pexels`, `Jina Reader`, `OmniVoice`, `MOSS TTS`, `Edge TTS`, `Wav2Lip`) in any user-facing HTML, JavaScript status message, or AJAX error.
   - Always run strings through `sanitize_public_status_text()` (PHP) and `sanitizeStatusText()` (JavaScript).
2. **Step 1 Lightweight Payload Rule:**
   - Never append `avatar`, `voice_ref`, or `brand_logo` file inputs to the Step 1 `ailg_draft_timeline` AJAX request. Those heavy binary files are exclusively used in Step 2->3 rendering (`ailg_start_runpod_render`).
3. **Deterministic Timeline Fallback:**
   - Step 1 drafting must never throw a hard error. If external LLM APIs fail or time out, `build_smart_fallback_timeline()` must structure a complete, multi-block timeline deterministically from the user's prompt, document, URL, or audio.
4. **POSIX Forward Slashes in ZIP Packages:**
   - When packaging the plugin (`ai-lesson-generator-1.zip` or `v7.6/ai-lesson-generator.zip`), always ensure all file entries inside the archive use POSIX forward slashes (`/`) and are placed under the root directory `ai-lesson-generator/`. Exclude nested `.zip` files during archive creation.
5. **Permanent URL Protocol:**
   - Never serve raw temporary S3/R2 presigned URLs directly to user download buttons. Always route through `admin-ajax.php?action=ailg_download_media&job_id={id}&type={type}` so download links remain valid permanently.
