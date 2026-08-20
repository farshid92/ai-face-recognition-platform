import { useEffect, useRef, useState, type DragEvent } from "react";

type VerifyResult = {
    name: string | null;
    score: number | null;
    is_match: boolean;
};

type Notice = {
    kind: "success" | "error" | "info";
    message: string;
};

type InputMode = "upload" | "camera";

const API = "";

function App() {
    const [mode, setMode] = useState<InputMode>("upload");
    const [file, setFile] = useState<File | null>(null);
    const [name, setName] = useState("");
    const [preview, setPreview] = useState("");
    const [result, setResult] = useState<VerifyResult | null>(null);
    const [notice, setNotice] = useState<Notice>({
        kind: "info",
        message: "Choose a clear, front-facing image to begin.",
    });
    const [busy, setBusy] = useState(false);
    const [apiOnline, setApiOnline] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    useEffect(() => {
        fetch(`${API}/health`)
            .then((response) => setApiOnline(response.ok))
            .catch(() => setApiOnline(false));
    }, []);

    useEffect(() => {
        if (!file) {
            setPreview("");
            return;
        }

        const url = URL.createObjectURL(file);
        setPreview(url);
        return () => URL.revokeObjectURL(url);
    }, [file]);

    function stopCamera() {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
    }

    useEffect(() => stopCamera, []);

    async function startCamera() {
        setFile(null);
        setResult(null);
        setMode("camera");

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            setNotice({ kind: "info", message: "Camera ready. Capture a frame when positioned." });
        } catch {
            setMode("upload");
            setNotice({
                kind: "error",
                message: "Could not access the camera. Check browser permissions.",
            });
        }
    }

    function switchToUpload() {
        stopCamera();
        setMode("upload");
        setNotice({ kind: "info", message: "Choose a clear, front-facing image to begin." });
    }

    function resetCapture(wasCameraMode: boolean) {
        setFile(null);
        if (inputRef.current) inputRef.current.value = "";

        if (wasCameraMode) {
            void startCamera();
        }
    }

    function captureFrame() {
        const video = videoRef.current;
        if (!video || !video.videoWidth) return;

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d")?.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            if (!blob) return;
            chooseFile(new File([blob], "capture.jpg", { type: "image/jpeg" }));
        }, "image/jpeg", 0.92);
    }

    function chooseFile(nextFile: File | undefined) {
        if (!nextFile) return;
        if (!nextFile.type.startsWith("image/")) {
            setNotice({ kind: "error", message: "Please choose an image file." });
            return;
        }

        stopCamera();
        setMode("upload");
        setFile(nextFile);
        setResult(null);
        setNotice({ kind: "info", message: "Image ready for analysis." });
    }

    function handleDrop(event: DragEvent<HTMLDivElement>) {
        event.preventDefault();
        chooseFile(event.dataTransfer.files[0]);
    }

    async function sendImage(path: string, form: FormData) {
        setBusy(true);
        setResult(null);

        try {
            const response = await fetch(`${API}${path}`, {
                method: "POST",
                body: form,
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.detail ?? "Request failed.");
            return payload;
        } catch (error) {
            setNotice({
                kind: "error",
                message: error instanceof Error ? error.message : "Request failed.",
            });
            return null;
        } finally {
            setBusy(false);
        }
    }

    async function verify() {
        if (!file) {
            setNotice({ kind: "error", message: "Choose an image first." });
            return;
        }

        const wasCameraMode = mode === "camera";
        const form = new FormData();
        form.append("image", file);
        const response = (await sendImage("/api/v1/verify", form)) as VerifyResult | null;
        if (!response) return;

        setResult(response);
        setNotice({
            kind: response.is_match ? "success" : "info",
            message: response.is_match
                ? `Identity confirmed as ${response.name}. Ready for the next person.`
                : "No registered identity passed the match threshold. Ready for the next person.",
        });
        resetCapture(wasCameraMode);
    }

    async function register() {
        if (!file || !name.trim()) {
            setNotice({ kind: "error", message: "Choose an image and enter a name." });
            return;
        }

        const wasCameraMode = mode === "camera";
        const form = new FormData();
        form.append("name", name.trim());
        form.append("image", file);
        const response = await sendImage("/api/v1/faces", form);
        if (!response) return;

        setNotice({
            kind: "success",
            message: `${response.name} is now registered. Ready for the next person.`,
        });
        setName("");
        resetCapture(wasCameraMode);
    }

    return (
        <main className="app-shell">
            <nav className="topbar">
                <a className="brand" href="/">
                    <span className="brand-mark">FR</span>
                    <span>Face Recognition <strong>Platform</strong></span>
                </a>
                <span className={`api-state ${apiOnline ? "online" : "offline"}`}>
                    <span className="state-dot" />
                    API {apiOnline ? "online" : "offline"}
                </span>
            </nav>

            <section className="intro">
                <div>
                    <p className="eyebrow">VISION LAB / MODULE 09</p>
                    <h1>Identify a face.<br /><em>Understand the signal.</em></h1>
                    <p className="intro-copy">
                        A focused interface for testing the SCRFD, ArcFace, and Qdrant pipeline.
                        Upload one image, then register or verify it against your local gallery.
                    </p>
                </div>
                <div className="metric-chip">
                    <span className="metric-value">512</span>
                    <span className="metric-label">embedding<br />dimensions</span>
                </div>
            </section>

            <section className="step-guide">
                <div className={`step-card ${!file ? "active" : "done"}`}>
                    <span className="step-index">1</span>
                    <div>
                        <p>Add a photo</p>
                        <span>Use Camera to capture, or Upload a file. One visible face.</span>
                    </div>
                </div>
                <div className={`step-card ${file ? "active" : ""}`}>
                    <span className="step-index">2</span>
                    <div>
                        <p>Register or verify</p>
                        <span>New person? Enter a name and Register. Seen before? Just Verify.</span>
                    </div>
                </div>
            </section>

            <section className="workspace">
                <div className="input-column">
                    <div className="mode-toggle">
                        <button
                            className={mode === "upload" ? "active" : ""}
                            onClick={switchToUpload}
                        >
                            Upload
                        </button>
                        <button
                            className={mode === "camera" ? "active" : ""}
                            onClick={startCamera}
                        >
                            Camera
                        </button>
                    </div>

                    {mode === "camera" ? (
                        <div className="drop-zone camera-zone">
                            <video ref={videoRef} muted playsInline className="camera-feed" />
                            <button className="capture-button" onClick={captureFrame}>
                                Capture
                            </button>
                        </div>
                    ) : (
                        <div
                            className={`drop-zone ${preview ? "has-image" : ""}`}
                            onDragOver={(event) => event.preventDefault()}
                            onDrop={handleDrop}
                            onClick={() => inputRef.current?.click()}
                        >
                            {preview ? (
                                <img className="preview" src={preview} alt="Selected face" />
                            ) : (
                                <div className="drop-content">
                                    <span className="upload-glyph">+</span>
                                    <p className="drop-title">Drop an image here</p>
                                    <p className="drop-hint">or click to browse · JPG, PNG</p>
                                </div>
                            )}
                            <input
                                ref={inputRef}
                                type="file"
                                accept="image/*"
                                onChange={(event) => chooseFile(event.target.files?.[0])}
                            />
                            {preview && <span className="change-image">Change image</span>}
                        </div>
                    )}
                </div>

                <aside className="control-panel">
                    <div className="panel-heading">
                        <span className="step-number">02</span>
                        <div>
                            <p className="panel-kicker">ACTION</p>
                            <h2>Register or verify</h2>
                        </div>
                    </div>
                    <p className="panel-description">
                        Enter a name only if this is a new person. Leave it blank and click
                        Verify to check against everyone already registered.
                    </p>

                    <label className="field-label" htmlFor="name">Registration name (new person only)</label>
                    <input
                        id="name"
                        className="text-input"
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        placeholder="e.g. Farshid"
                        disabled={!file}
                    />

                    <div className="action-group">
                        <button className="button primary" onClick={verify} disabled={busy || !file}>
                            {busy ? "Processing..." : "Verify face"}
                            <span>→</span>
                        </button>
                        <button className="button secondary" onClick={register} disabled={busy || !file}>
                            Register identity
                            <span>+</span>
                        </button>
                    </div>

                    <div className={`notice ${notice.kind}`}>
                        <span className="notice-mark">{notice.kind === "error" ? "!" : "i"}</span>
                        <span>{notice.message}</span>
                    </div>
                </aside>
            </section>

            {result && (
                <section className={`result-bar ${result.is_match ? "match" : "unknown"}`}>
                    <div>
                        <span className="result-kicker">VERIFICATION RESULT</span>
                        <strong>{result.is_match ? result.name : "Unknown identity"}</strong>
                    </div>
                    <div className="score">
                        <span>Cosine similarity</span>
                        <strong>{result.score === null ? "--" : result.score.toFixed(4)}</strong>
                    </div>
                </section>
            )}

            <footer>
                <span>LOCAL INFERENCE WORKBENCH</span>
                <span>SCRFD · ARCFACE · QDRANT</span>
            </footer>
        </main>
    );
}

export default App;
