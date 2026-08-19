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

const API = "";

function App() {
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

    function chooseFile(nextFile: File | undefined) {
        if (!nextFile) return;
        if (!nextFile.type.startsWith("image/")) {
            setNotice({ kind: "error", message: "Please choose an image file." });
            return;
        }

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

        const form = new FormData();
        form.append("image", file);
        const response = (await sendImage("/api/v1/verify", form)) as VerifyResult | null;
        if (!response) return;

        setResult(response);
        setNotice({
            kind: response.is_match ? "success" : "info",
            message: response.is_match
                ? `Identity confirmed as ${response.name}.`
                : "No registered identity passed the match threshold.",
        });
    }

    async function register() {
        if (!file || !name.trim()) {
            setNotice({ kind: "error", message: "Choose an image and enter a name." });
            return;
        }

        const form = new FormData();
        form.append("name", name.trim());
        form.append("image", file);
        const response = await sendImage("/api/v1/faces", form);
        if (!response) return;

        setNotice({ kind: "success", message: `${response.name} is now registered.` });
        setName("");
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

            <section className="workspace">
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

                <aside className="control-panel">
                    <div className="panel-heading">
                        <span className="step-number">01</span>
                        <div>
                            <p className="panel-kicker">INPUT</p>
                            <h2>Face sample</h2>
                        </div>
                    </div>
                    <p className="panel-description">
                        Use a single visible face. Multiple faces are rejected to keep each
                        registration and verification decision unambiguous.
                    </p>

                    <label className="field-label" htmlFor="name">Registration name</label>
                    <input
                        id="name"
                        className="text-input"
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        placeholder="e.g. Farshid"
                    />

                    <div className="action-group">
                        <button className="button primary" onClick={verify} disabled={busy}>
                            {busy ? "Processing..." : "Verify face"}
                            <span>→</span>
                        </button>
                        <button className="button secondary" onClick={register} disabled={busy}>
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
