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

### Weights (Hugging Face)

The plugin pulls its weights from Hugging Face at deploy time and requires a token. Create the Modal secret it reads:

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
```

Without `HF_TOKEN` the weight download fails.
