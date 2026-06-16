# tongflow-modal-infinitetalk

Official TongFlow plugin. Audio-driven lip-sync with **InfiniteTalk** (`MeiGen-AI/InfiniteTalk`, on `Wan-AI/Wan2.1-I2V-14B-480P` with a wav2vec audio encoder), running on a GPU via [Modal](https://modal.com). Turns an audio track plus a video (or image) into a talking-head video.

## Capabilities

- **Lip sync** (`audio-video-lip-sync`) — drive a subject's lips from an audio track (audio + video → video).

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | ✅ | Required to fetch the Wan2.1 / InfiniteTalk weights from Hugging Face. |

### Weights (Hugging Face)

The plugin injects `HF_TOKEN` from your TongFlow Settings into the Modal download job at deploy time — no manual `modal secret create` needed. Without it the weight download fails.
