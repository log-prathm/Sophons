export interface RecordingResult {
  blob: Blob;
  speechStartedAtMs: number;
  speechEndedAtMs: number;
  durationMs: number;
}

interface RecorderOptions {
  silenceDurationMs: number;
  silenceThreshold: number;
  minSpeechMs: number;
  onLevel?: (level: number) => void;
  onListeningStarted?: () => void;
  onSpeechStart?: (speechStartedAtMs: number) => void;
  onSegmentReady: (segment: RecordingResult) => void | Promise<void>;
}

function floatTo16BitPCM(view: DataView, offset: number, input: Float32Array): void {
  for (let index = 0; index < input.length; index += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeText = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPCM(view, 44, samples);

  return new Blob([buffer], { type: "audio/wav" });
}

function mergeBuffers(chunks: Float32Array[], totalLength: number): Float32Array {
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function computeRms(input: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < input.length; index += 1) {
    sum += input[index] * input[index];
  }
  return Math.sqrt(sum / input.length);
}

export class AutoSilenceRecorder {
  private audioContext: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private sink: GainNode | null = null;
  private options: RecorderOptions | null = null;
  private listening = false;
  private speechStartedAtMs: number | null = null;
  private lastVoiceAtMs = 0;
  private chunks: Float32Array[] = [];
  private totalLength = 0;
  private preRoll: Float32Array[] = [];
  private preRollLength = 0;
  private readonly maxPreRollChunks = 6;

  async start(options: RecorderOptions): Promise<void> {
    this.options = options;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 16000,
      },
    });

    this.audioContext = new AudioContext();
    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    this.sink = this.audioContext.createGain();
    this.sink.gain.value = 0;

    this.source.connect(this.processor);
    this.processor.connect(this.sink);
    this.sink.connect(this.audioContext.destination);

    this.listening = true;
    this.chunks = [];
    this.totalLength = 0;
    this.preRoll = [];
    this.preRollLength = 0;
    this.speechStartedAtMs = null;
    this.lastVoiceAtMs = 0;

    this.processor.onaudioprocess = (event) => {
      if (!this.listening || !this.options) {
        return;
      }

      const channel = event.inputBuffer.getChannelData(0);
      const chunk = new Float32Array(channel);
      const now = performance.now();
      const level = computeRms(chunk);
      this.options.onLevel?.(level);

      this.preRoll.push(chunk);
      this.preRollLength += chunk.length;
      if (this.preRoll.length > this.maxPreRollChunks) {
        const removed = this.preRoll.shift();
        this.preRollLength -= removed?.length ?? 0;
      }

      if (level >= this.options.silenceThreshold) {
        if (this.speechStartedAtMs === null) {
          this.speechStartedAtMs = now;
          this.lastVoiceAtMs = now;
          this.chunks = [...this.preRoll];
          this.totalLength = this.preRollLength;
          this.options.onSpeechStart?.(now);
        } else {
          this.lastVoiceAtMs = now;
        }
        this.chunks.push(chunk);
        this.totalLength += chunk.length;
        return;
      }

      if (this.speechStartedAtMs === null) {
        return;
      }

      this.chunks.push(chunk);
      this.totalLength += chunk.length;
      const silenceElapsed = now - this.lastVoiceAtMs;
      const speechElapsed = now - this.speechStartedAtMs;
      if (
        silenceElapsed >= this.options.silenceDurationMs &&
        speechElapsed >= this.options.minSpeechMs
      ) {
        void this.emitSegment(now);
      }
    };

    this.options.onListeningStarted?.();
  }

  async cancel(): Promise<void> {
    await this.cleanup();
  }

  private async emitSegment(speechEndedAtMs: number): Promise<void> {
    if (!this.options || this.speechStartedAtMs === null || this.totalLength === 0) {
      await this.cleanup();
      return;
    }

    const sampleRate = this.audioContext?.sampleRate ?? 16000;
    const merged = mergeBuffers(this.chunks, this.totalLength);
    const blob = encodeWav(merged, sampleRate);
    const result: RecordingResult = {
      blob,
      speechStartedAtMs: this.speechStartedAtMs,
      speechEndedAtMs,
      durationMs: speechEndedAtMs - this.speechStartedAtMs,
    };

    const callback = this.options.onSegmentReady;
    await this.cleanup();
    await callback(result);
  }

  private async cleanup(): Promise<void> {
    this.listening = false;
    this.processor?.disconnect();
    this.source?.disconnect();
    this.sink?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());

    if (this.audioContext) {
      await this.audioContext.close();
    }

    this.audioContext = null;
    this.stream = null;
    this.processor = null;
    this.source = null;
    this.sink = null;
    this.options = null;
    this.chunks = [];
    this.totalLength = 0;
    this.preRoll = [];
    this.preRollLength = 0;
    this.speechStartedAtMs = null;
    this.lastVoiceAtMs = 0;
  }
}
