import { useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, FormEvent, ReactElement } from 'react';
import githubMark from './assets/github.svg';

const MAX_AUDIO_SIZE = 100 * 1024 * 1024;

interface Prediction {
  predicted_genre: string;
  probabilities: Record<string, number>;
}

/** Check a selected file's extension and size; return an English error or an empty string. */
function validateFile(file: File): string {
  if (!/\.wav$/i.test(file.name)) {
    return 'Please select a WAV file.';
  }
  if (file.size === 0) {
    return 'This file is empty. Please select a valid WAV file.';
  }
  if (file.size > MAX_AUDIO_SIZE) {
    return 'File too large. Please select a WAV file of 100 MiB or less.';
  }
  return '';
}

/** POST to an endpoint with an optional file body; return prediction or a readable error. */
async function requestPrediction(
  endpoint: string,
  body?: FormData,
): Promise<Prediction> {
  let response: Response;
  try {
    response = await fetch(endpoint, { method: 'POST', ...(body ? { body } : {}) });
  } catch {
    throw new Error(
      'Could not connect to the API. Check that FastAPI is running and try again.',
    );
  }

  if (!response.ok) {
    const errorBody = (await response.json().catch((): null => null)) as {
      detail?: unknown;
    } | null;
    throw new Error(
      typeof errorBody?.detail === 'string'
        ? errorBody.detail
        : 'Could not analyze this audio. Please try again.',
    );
  }

  return response.json() as Promise<Prediction>;
}

/** Render audio and YouTube forms with shared results; takes no props. */
export default function App(): ReactElement {
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState('');
  const [processing, setProcessing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);

  /** Validate a selected file or null; clear previous feedback and return nothing. */
  function selectFile(selectedFile: File | null): void {
    if (processing) return;
    const validationError = selectedFile ? validateFile(selectedFile) : '';
    setFile(validationError ? null : selectedFile);
    setError(validationError);
    setPrediction(null);
  }

  /** Read a file-input change event; validate a choice or preserve it on cancellation. */
  function handleFileChange(event: ChangeEvent<HTMLInputElement>): void {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) selectFile(selectedFile);
  }

  /** Open the hidden file picker; takes no arguments and preserves the current selection. */
  function openFilePicker(): void {
    if (processing || !fileInput.current) return;
    fileInput.current.value = '';
    fileInput.current.click();
  }

  /** Track a file entering the drop zone; highlight it unless processing. */
  function handleDragEnter(event: DragEvent<HTMLButtonElement>): void {
    event.preventDefault();
    if (processing || !event.dataTransfer.types.includes('Files')) return;
    dragDepth.current += 1;
    setDragging(true);
  }

  /** Allow a drag-over event to drop files; indicate whether selection is available. */
  function handleDragOver(event: DragEvent<HTMLButtonElement>): void {
    event.preventDefault();
    event.dataTransfer.dropEffect = processing ? 'none' : 'copy';
  }

  /** Track a drag leaving the zone or its children; clear the highlight on exit. */
  function handleDragLeave(event: DragEvent<HTMLButtonElement>): void {
    event.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  }

  /** Read a drop event; validate one file without automatically starting analysis. */
  function handleDrop(event: DragEvent<HTMLButtonElement>): void {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (processing || event.dataTransfer.files.length === 0) return;

    if (fileInput.current) fileInput.current.value = '';
    if (event.dataTransfer.files.length > 1) {
      selectFile(null);
      setError('Please drop one WAV file at a time.');
      return;
    }
    selectFile(event.dataTransfer.files[0]);
  }

  /** Request prediction for endpoint/body; update shared loading and feedback state. */
  async function analyze(endpoint: string, body?: FormData): Promise<void> {
    if (processing) return;
    setProcessing(true);
    setError('');
    setPrediction(null);
    try {
      setPrediction(await requestPrediction(endpoint, body));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Something went wrong. Please try again.',
      );
    } finally {
      setProcessing(false);
    }
  }

  /** Submit the selected WAV from event; await analysis without reloading the page. */
  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!file || processing) return;
    const body = new FormData();
    body.append('file', file);
    await analyze('/api/predict-audio', body);
  }

  /** Read a URL change event; clear old feedback while preserving the selected WAV. */
  function handleUrlChange(event: ChangeEvent<HTMLInputElement>): void {
    if (processing) return;
    setYoutubeUrl(event.target.value);
    setError('');
    setPrediction(null);
  }

  /** Validate event's URL form and submit an encoded query; await shared analysis. */
  async function handleYoutubeSubmit(
    event: FormEvent<HTMLFormElement>,
  ): Promise<void> {
    event.preventDefault();
    if (processing) return;
    const url = youtubeUrl.trim();
    try {
      const parsed = new URL(url);
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error();
    } catch {
      setPrediction(null);
      setError('Please enter a valid HTTP or HTTPS YouTube URL.');
      return;
    }
    await analyze(`/api/predict-youtube?${new URLSearchParams({ url })}`);
  }

  const probabilities = Object.entries(prediction?.probabilities ?? {}).sort(
    (first, second): number => second[1] - first[1],
  );

  return (
    <>
      <main>
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-content">
            <p className="eyebrow">Musical Machine Learning</p>
            <h1 id="page-title">Music Genre Identifier</h1>
            <p className="hero-description">
              What does your music sound like to a machine learning model?
            </p>
            <a
              className="repo-button"
              href="https://github.com/GaryVidalC/music-genre-identifier"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img src={githubMark} alt="" width="20" height="20" />
              GitHub Repo
            </a>
          </div>
        </section>

        <div className="workspace">
          <section className="card upload-card" aria-labelledby="upload-title">
            <p className="section-label">01 / Your audio</p>
            <h2 id="upload-title">Start with a track</h2>
            <p className="muted">
              Choose an audio file from your device. Only WAV files are
              supported.
            </p>

            <form onSubmit={handleSubmit} aria-busy={processing}>
              <button
                type="button"
                className={dragging ? 'file-picker dragging' : 'file-picker'}
                disabled={processing}
                onClick={openFilePicker}
                onDragEnter={handleDragEnter}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <span className="music-note" aria-hidden="true">
                  ♫
                </span>
                <span className="file-picker-title" title={file?.name}>
                  {file ? file.name : 'Choose a WAV file'}
                </span>
                {!file && (
                  <span className="file-help">
                    Click to browse or drag &amp; drop a file
                  </span>
                )}
              </button>
              <input
                ref={fileInput}
                id="audio-file"
                name="file"
                type="file"
                accept=".wav,audio/wav,audio/x-wav"
                aria-label="Choose a WAV file"
                hidden
                disabled={processing}
                onChange={handleFileChange}
              />

              <button
                className="analyze-button"
                type="submit"
                disabled={!file || processing}
              >
                {processing ? 'Analyzing…' : 'Analyze audio'}
              </button>
            </form>

            <form
              className="youtube-form"
              onSubmit={handleYoutubeSubmit}
              aria-busy={processing}
              noValidate
            >
              <label htmlFor="youtube-url">YouTube URL</label>
              <input
                id="youtube-url"
                name="url"
                type="url"
                placeholder="https://www.youtube.com/watch?v=…"
                value={youtubeUrl}
                onChange={handleUrlChange}
                disabled={processing}
                required
              />
              <button
                className="analyze-button"
                type="submit"
                disabled={!youtubeUrl.trim() || processing}
              >
                Analyze YouTube
              </button>
            </form>

            {processing && (
              <p className="feedback muted" role="status">
                Processing your audio. Longer tracks may take a little more
                time.
              </p>
            )}
            {error && (
              <p className="feedback error" role="alert">
                {error}
              </p>
            )}

            <p className="audio-note">
              The model analyzes 30-second segments and averages predictions
              across up to ten segments for longer tracks.
            </p>
          </section>

          <section
            className="card results-card"
            aria-labelledby="results-title"
            aria-live="polite"
            aria-busy={processing}
          >
            <p className="section-label">02 / The prediction</p>
            <h2 id="results-title">Your audio analysis</h2>

            {prediction ? (
              <>
                <div className="genre-result">
                  <p className="muted">Predicted genre</p>
                  <h3>{prediction.predicted_genre}</h3>
                </div>
                <p className="probabilities-label">Other possible genres</p>
                <ul className="probability-list">
                  {probabilities.map(([genre, probability]): ReactElement => (
                    <li
                      key={genre}
                      className={
                        genre === prediction.predicted_genre
                          ? 'probability-row winner'
                          : 'probability-row'
                      }
                    >
                      <div className="probability-heading">
                        <span className="genre-label">{genre}</span>
                        <span>{(probability * 100).toFixed(1)}%</span>
                      </div>
                      <div
                        className="probability-track"
                        role="meter"
                        aria-label={`${genre} probability`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={probability * 100}
                      >
                        <span style={{ width: `${probability * 100}%` }} />
                      </div>
                    </li>
                  ))}
                </ul>
                <p className="muted result-note">
                  This is just a genre prediction, not an objective label.
                </p>
              </>
            ) : (
              <div className="empty-result">
                <div className="equalizer" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                </div>
                <p>
                  {processing
                    ? 'Finding the patterns in your audio…'
                    : 'Your results will appear here.'}
                </p>
                <p className="muted">
                  Predictions across ten musical genres.
                </p>
              </div>
            )}
          </section>
        </div>
      </main>

      <footer className="footer">
        <span>
          Built by <a href="https://gvidal.cl">Gary Vidal</a>
        </span>
        <span>Traditional Machine learning.</span>
      </footer>
    </>
  );
}
