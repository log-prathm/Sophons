from __future__ import annotations

from importlib import import_module
import os
import tempfile
from pathlib import Path
import re

import numpy as np
import soundfile as sf

from backend.app.core.runtime_env import load_runtime_env

load_runtime_env()

import torch

from backend.app.core.schemas import ModelManifest
from backend.app.pipeline.base import TTSEngine
from backend.app.pipeline.device_utils import normalize_device_name, torch_cuda_available


class MeloTTSEngine(TTSEngine):
    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._prime_unidic()
        self._prime_local_hf_aliases()
        try:
            from melo.api import TTS
            from huggingface_hub import hf_hub_download
            from melo.download_utils import LANG_TO_HF_REPO_ID
        except Exception as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "Melo TTS is currently experimental in this shared environment. "
                "The main stable GPU TTS path is Kokoro."
            ) from exc

        config = manifest.config
        self.language = config.get("language", "EN")
        self.speed = float(config.get("speed", 0.95))
        self.speaker = config.get("speaker", "EN_INDIA")
        self.disable_bert = bool(config.get("disable_bert", True))
        self.half_precision = bool(config.get("half_precision", True))
        self.sdp_ratio = float(config.get("sdp_ratio", 0.2))
        self.noise_scale = float(config.get("noise_scale", 0.6))
        self.noise_scale_w = float(config.get("noise_scale_w", 0.8))
        self.sentence_pause_ms = int(config.get("sentence_pause_ms", 40))
        requested_device = normalize_device_name(config.get("device", "gpu"), default="gpu")
        if requested_device == "cuda" and not torch_cuda_available():
            raise RuntimeError(
                "GPU was selected for Melo, but PyTorch CUDA is not available in this environment."
            )
        device = "cuda" if requested_device == "cuda" else "cpu"
        repo_id = LANG_TO_HF_REPO_ID[self.language.split("-")[0].upper()]
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", local_files_only=True)
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pth", local_files_only=True)
        self.engine = TTS(
            language=self.language,
            device=device,
            config_path=config_path,
            ckpt_path=ckpt_path,
        )
        if device == "cuda" and self.half_precision:
            self.engine.model = self.engine.model.half()
        self.engine.hps.data.disable_bert = self.disable_bert
        self._melo_api = import_module(self.engine.__class__.__module__)
        self.speaker_ids = self.engine.hps.data.spk2id
        if self.speaker not in self.speaker_ids:
            normalized = self.speaker.replace("-", "_").upper()
            if normalized in self.speaker_ids:
                self.speaker = normalized
            elif "EN_INDIA" in self.speaker_ids:
                self.speaker = "EN_INDIA"
            else:
                self.speaker = next(iter(self.speaker_ids.keys()))

    def synthesize(self, text: str, output_path: Path) -> Path:
        prepared_text = _compact_tts_text(text)
        chunks = self.engine.split_sentences_into_pieces(prepared_text, self.engine.language, quiet=True)
        audio_segments = []
        utils = self._melo_api.utils
        pause_samples = int((self.engine.hps.data.sampling_rate * (self.sentence_pause_ms / 1000)) / max(self.speed, 0.1))

        with torch.inference_mode():
            for chunk in chunks:
                if self.engine.language in ["EN", "ZH_MIX_EN"]:
                    chunk = re.sub(r"([a-z])([A-Z])", r"\1 \2", chunk)
                bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
                    chunk,
                    self.engine.language,
                    self.engine.hps,
                    self.engine.device,
                    self.engine.symbol_to_id,
                )
                x_tst = phones.to(self.engine.device).unsqueeze(0)
                tones = tones.to(self.engine.device).unsqueeze(0)
                lang_ids = lang_ids.to(self.engine.device).unsqueeze(0)
                bert = bert.to(self.engine.device).unsqueeze(0)
                ja_bert = ja_bert.to(self.engine.device).unsqueeze(0)
                if self.engine.device == "cuda" and self.half_precision:
                    bert = bert.half()
                    ja_bert = ja_bert.half()
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.engine.device)
                speakers = torch.LongTensor([self.speaker_ids[self.speaker]]).to(self.engine.device)
                audio = self.engine.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=self.sdp_ratio,
                    noise_scale=self.noise_scale,
                    noise_scale_w=self.noise_scale_w,
                    length_scale=1.0 / self.speed,
                )[0][0, 0].data.cpu().float().numpy()
                audio_segments.append(audio)

        merged = _concat_audio(audio_segments, pause_samples)
        sf.write(output_path, merged, self.engine.hps.data.sampling_rate)
        return output_path

    def warmup(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            self.synthesize("Ready.", Path(handle.name))

    def _prime_unidic(self) -> None:
        try:
            import unidic_lite
        except Exception:
            return

        dic_dir = Path(unidic_lite.DICDIR)
        if dic_dir.exists():
            os.environ.setdefault("MECABRC", str(dic_dir / "mecabrc"))
            os.environ.setdefault("UNIDICDIR", str(dic_dir))

    def _prime_local_hf_aliases(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        project_root = Path(__file__).resolve().parents[4]
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        aliases = {
            "bert-base-uncased": cache_root / "models--bert-base-uncased" / "snapshots",
            "bert-base-multilingual-uncased": cache_root / "models--bert-base-multilingual-uncased" / "snapshots",
        }

        for alias, snapshots_root in aliases.items():
            if (project_root / alias).exists() or not snapshots_root.exists():
                continue
            snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
            if not snapshots:
                continue
            try:
                (project_root / alias).symlink_to(snapshots[-1], target_is_directory=True)
            except FileExistsError:
                continue
            except OSError:
                continue


def _compact_tts_text(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = normalized.replace("•", ", ").replace("*", "")
    return normalized.strip()


def _concat_audio(audio_segments: list, pause_samples: int) -> np.ndarray:
    merged: list[float] = []
    for index, segment in enumerate(audio_segments):
        merged.extend(segment.reshape(-1).tolist())
        if index < len(audio_segments) - 1 and pause_samples > 0:
            merged.extend([0.0] * pause_samples)
    return np.asarray(merged, dtype=np.float32)
