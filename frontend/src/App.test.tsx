import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import App from './App';

const prediction = {
  predicted_genre: 'rock',
  probabilities: { blues: 0.25, rock: 0.75 },
};

/** Restore the DOM after a test; takes no arguments and returns nothing. */
function cleanUp(): void {
  cleanup();
}
afterEach(cleanUp);

/** Select a supplied file in the rendered upload input; returns nothing. */
function selectFile(file: File): void {
  fireEvent.change(screen.getByLabelText('Choose a WAV file'), {
    target: { files: [file] },
  });
}

/** Drop supplied files onto the upload zone; return nothing. */
function dropFiles(...files: File[]): void {
  const dropZone = screen.getByLabelText('Choose a WAV file')
    .parentElement!.querySelector('.file-picker')!;
  expect(
    fireEvent.drop(dropZone, { dataTransfer: { files, types: ['Files'] } }),
  ).toBe(false);
}

/** Build a successful JSON response containing the API prediction; returns the response. */
function predictionResponse(): Response {
  return new Response(JSON.stringify(prediction), {
    headers: { 'Content-Type': 'application/json' },
  });
}

/** Verify navigation, picker replacement and upload; takes no arguments and completes asynchronously. */
test('links to the repo and supports dropping files without changing the upload flow', async (): Promise<void> => {
  let resolveRequest!: (response: Response) => void;
  const request = vi.fn(
    (): Promise<Response> =>
      new Promise((resolve): void => {
        resolveRequest = resolve;
      }),
  );
  vi.stubGlobal('fetch', request);
  render(<App />);
  const repoLink = screen.getByRole('link', { name: 'GitHub Repo' });
  expect(repoLink.getAttribute('href')).toBe(
    'https://github.com/GaryVidalC/music-genre-identifier',
  );
  expect(repoLink.getAttribute('target')).toBe('_blank');
  expect(repoLink.getAttribute('rel')).toBe('noopener noreferrer');
  expect(screen.queryByRole('banner')).toBeNull();
  expect(screen.queryByRole('navigation')).toBeNull();
  const file = new File(['audio'], 'track.wav', { type: 'audio/wav' });
  const input = screen.getByLabelText('Choose a WAV file') as HTMLInputElement;
  const dropZone = screen.getByRole('button', { name: /Choose a WAV file/ });
  expect(input.hidden).toBe(true);
  expect(screen.getByText('Click to browse or drag & drop a file')).toBeTruthy();
  const openPicker = vi.spyOn(input, 'click').mockImplementation((): void => {});
  fireEvent.click(dropZone);
  expect(openPicker).toHaveBeenCalledTimes(1);
  selectFile(new File(['previous'], 'previous.wav'));
  expect(dropZone.textContent).toContain('previous.wav');
  const dragData = { dataTransfer: { files: [file], types: ['Files'] } };
  fireEvent.dragEnter(dropZone, dragData);
  expect(dropZone.classList.contains('dragging')).toBe(true);
  fireEvent.dragEnter(dropZone.querySelector('.music-note')!, dragData);
  fireEvent.dragLeave(dropZone.querySelector('.music-note')!, dragData);
  expect(dropZone.classList.contains('dragging')).toBe(true);
  fireEvent.dragLeave(dropZone, dragData);
  expect(dropZone.classList.contains('dragging')).toBe(false);
  fireEvent.dragEnter(dropZone, dragData);
  expect(fireEvent.dragOver(dropZone, dragData)).toBe(false);
  dropFiles(file);
  expect(dropZone.classList.contains('dragging')).toBe(false);
  expect(screen.getByText('track.wav')).toBeTruthy();
  expect(dropZone.textContent).toContain('track.wav');
  expect(screen.getByText('track.wav').getAttribute('title')).toBe('track.wav');
  expect(screen.queryByText('Click to browse or drag & drop a file')).toBeNull();
  fireEvent.click(dropZone);
  expect(openPicker).toHaveBeenCalledTimes(2);
  fireEvent.change(input, { target: { files: [] } });
  fireEvent(input, new Event('cancel'));
  expect(dropZone.textContent).toContain('track.wav');
  expect(request).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Analyze audio' }));
  expect(
    (screen.getByRole('button', { name: 'Analyzing…' }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  expect((dropZone as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(dropZone);
  expect(openPicker).toHaveBeenCalledTimes(2);
  expect(
    (screen.getByLabelText('Choose a WAV file') as HTMLInputElement).disabled,
  ).toBe(true);
  expect(screen.getByRole('status').textContent).toContain(
    'Processing your audio',
  );
  expect((screen.getByLabelText('YouTube URL') as HTMLInputElement).disabled)
    .toBe(true);
  expect((screen.getByRole('button', { name: 'Analyze YouTube' }) as HTMLButtonElement).disabled)
    .toBe(true);
  fireEvent.submit(
    screen.getByRole('button', { name: 'Analyzing…' }).closest('form')!,
  );
  const nextFile = new File(['other'], 'next.wav');
  fireEvent.dragEnter(dropZone, dragData);
  expect(dropZone.classList.contains('dragging')).toBe(false);
  dropFiles(nextFile);
  expect(screen.getByText('track.wav')).toBeTruthy();
  expect(screen.queryByText('next.wav')).toBeNull();
  expect(request).toHaveBeenCalledTimes(1);

  const [url, options] = request.mock.calls[0] as unknown as [
    string,
    RequestInit,
  ];
  expect(url).toBe('/api/predict-audio');
  expect(options.method).toBe('POST');
  expect((options.body as FormData).get('file')).toBe(file);
  expect(options.headers).toBeUndefined();

  await act(async (): Promise<void> => resolveRequest(predictionResponse()));
  expect(screen.getByRole('heading', { name: 'rock' })).toBeTruthy();
  expect(
    screen
      .getAllByRole('meter')
      .map((meter): string | null => meter.getAttribute('aria-label')),
  ).toEqual(['rock probability', 'blues probability']);
  expect(screen.getByText('75.0%')).toBeTruthy();

  dropFiles(nextFile);
  expect(dropZone.textContent).toContain('next.wav');
  expect(screen.queryByRole('heading', { name: 'rock' })).toBeNull();
  expect(screen.getByText('Your results will appear here.')).toBeTruthy();

  const youtubeInput = screen.getByLabelText('YouTube URL') as HTMLInputElement;
  const videoUrl = 'https://www.youtube.com/watch?v=I4v4-Mi6qZA&list=test';
  fireEvent.change(youtubeInput, { target: { value: `  ${videoUrl}  ` } });
  const youtubeForm = youtubeInput.closest('form')!;
  fireEvent.submit(youtubeForm);
  expect(youtubeInput.disabled).toBe(true);
  expect((dropZone as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole('button', { name: 'Analyzing…' }) as HTMLButtonElement).disabled)
    .toBe(true);
  fireEvent.submit(youtubeForm);
  fireEvent.submit(input.closest('form')!);
  dropFiles(file);
  fireEvent.change(youtubeInput, { target: { value: 'https://youtu.be/other' } });
  expect(youtubeInput.value.trim()).toBe(videoUrl);
  expect(dropZone.textContent).toContain('next.wav');
  expect(request).toHaveBeenCalledTimes(2);
  const [youtubeEndpoint, youtubeOptions] = request.mock.calls[1] as unknown as [
    string, RequestInit,
  ];
  expect(youtubeEndpoint).toBe(
    `/api/predict-youtube?${new URLSearchParams({ url: videoUrl })}`,
  );
  expect(youtubeOptions.method).toBe('POST');
  expect(youtubeOptions.body).toBeUndefined();
  await act(async (): Promise<void> => resolveRequest(predictionResponse()));
  expect(screen.getByRole('heading', { name: 'rock' })).toBeTruthy();
  expect(screen.getByText('75.0%')).toBeTruthy();
  fireEvent.change(youtubeInput, { target: { value: 'https://youtu.be/next' } });
  expect(screen.queryByRole('heading', { name: 'rock' })).toBeNull();
  openPicker.mockRestore();
});

/** Verify shared validation for selected or dropped files; takes no arguments and returns nothing. */
test('rejects invalid selections and drops before making a request', (): void => {
  const request = vi.fn();
  vi.stubGlobal('fetch', request);
  render(<App />);
  expect(
    (screen.getByRole('button', { name: 'Analyze audio' }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);

  selectFile(new File(['audio'], 'track.mp3'));
  expect(screen.getByRole('alert').textContent).toContain(
    'Please select a WAV file',
  );
  dropFiles(new File(['audio'], 'track.mp3'));
  expect(screen.getByRole('alert').textContent).toContain(
    'Please select a WAV file',
  );
  dropFiles(new File([], 'empty.wav'));
  expect(screen.getByRole('alert').textContent).toContain('This file is empty');

  const largeFile = new File(['audio'], 'large.wav');
  Object.defineProperty(largeFile, 'size', { value: 100 * 1024 * 1024 + 1 });
  dropFiles(largeFile);
  expect(screen.getByRole('alert').textContent).toContain('File too large');
  dropFiles(
    new File(['first'], 'first.wav'),
    new File(['second'], 'second.wav'),
  );
  expect(screen.getByRole('alert').textContent).toContain(
    'one WAV file at a time',
  );
  expect(request).not.toHaveBeenCalled();

  const validFile = new File(['audio'], 'track.WAV');
  Object.defineProperty(validFile, 'size', { value: 100 * 1024 * 1024 });
  selectFile(validFile);
  expect(screen.queryByRole('alert')).toBeNull();
  expect(
    (screen.getByRole('button', { name: 'Analyze audio' }) as HTMLButtonElement)
      .disabled,
  ).toBe(false);
  dropFiles();
  expect(screen.getByText('track.WAV')).toBeTruthy();

  const youtubeInput = screen.getByLabelText('YouTube URL');
  expect((screen.getByRole('button', { name: 'Analyze YouTube' }) as HTMLButtonElement).disabled)
    .toBe(true);
  for (const url of ['', 'not a URL', 'ftp://youtube.com/video']) {
    fireEvent.change(youtubeInput, { target: { value: url } });
    fireEvent.submit(youtubeInput.closest('form')!);
    expect(screen.getByRole('alert').textContent).toContain('valid HTTP or HTTPS');
  }
  fireEvent.change(youtubeInput, {
    target: { value: 'https://youtu.be/I4v4-Mi6qZA' },
  });
  expect(screen.queryByRole('alert')).toBeNull();
  expect(request).not.toHaveBeenCalled();
});

test('shows API and connection errors, and allows retrying the selected file', async (): Promise<void> => {
  const request = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          detail: 'Invalid audio file. Please upload a valid .wav file.',
        }),
        { status: 400 },
      ),
    )
    .mockResolvedValueOnce(new Response('Service unavailable', { status: 503 }))
    .mockRejectedValueOnce(new TypeError('Failed to fetch'))
    .mockResolvedValueOnce(predictionResponse())
    .mockResolvedValueOnce(new Response(
      JSON.stringify({ detail: 'Playlist URLs are not supported.' }),
      { status: 400 },
    ))
    .mockRejectedValueOnce(new TypeError('Failed to fetch'))
    .mockResolvedValueOnce(predictionResponse());
  vi.stubGlobal('fetch', request);
  render(<App />);
  selectFile(new File(['audio'], 'track.wav'));

  fireEvent.click(screen.getByRole('button', { name: 'Analyze audio' }));
  expect((await screen.findByRole('alert')).textContent).toContain(
    'Invalid audio file',
  );
  await waitFor((): void =>
    expect(
      (
        screen.getByRole('button', {
          name: 'Analyze audio',
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false),
  );

  fireEvent.click(screen.getByRole('button', { name: 'Analyze audio' }));
  await waitFor((): void =>
    expect(screen.getByRole('alert').textContent).toContain(
      'Could not analyze this audio',
    ),
  );
  fireEvent.click(screen.getByRole('button', { name: 'Analyze audio' }));
  await waitFor((): void =>
    expect(screen.getByRole('alert').textContent).toContain(
      'Could not connect to the API',
    ),
  );
  fireEvent.click(screen.getByRole('button', { name: 'Analyze audio' }));
  expect(await screen.findByRole('heading', { name: 'rock' })).toBeTruthy();
  expect(screen.queryByRole('alert')).toBeNull();
  expect(request).toHaveBeenCalledTimes(4);

  const youtubeInput = screen.getByLabelText('YouTube URL') as HTMLInputElement;
  const videoUrl = 'https://youtu.be/I4v4-Mi6qZA';
  fireEvent.change(youtubeInput, { target: { value: videoUrl } });
  expect(screen.queryByRole('heading', { name: 'rock' })).toBeNull();
  fireEvent.submit(youtubeInput.closest('form')!);
  expect((await screen.findByRole('alert')).textContent).toContain('Playlist URLs');
  expect(youtubeInput.value).toBe(videoUrl);
  fireEvent.submit(youtubeInput.closest('form')!);
  await waitFor((): void =>
    expect(screen.getByRole('alert').textContent).toContain('Could not connect'),
  );
  expect(youtubeInput.value).toBe(videoUrl);
  fireEvent.submit(youtubeInput.closest('form')!);
  expect(await screen.findByRole('heading', { name: 'rock' })).toBeTruthy();
  expect(screen.queryByRole('alert')).toBeNull();
  expect(request).toHaveBeenCalledTimes(7);
});
