const resumeInput = document.getElementById("resume-input");
const jobDescriptionInput = document.getElementById("jd-input");
const resumeDropZone = document.getElementById("resume-drop-zone");
const jobDescriptionDropZone = document.getElementById("jd-drop-zone");
const analyzeButton = document.getElementById("analyze-button");
const loadingSection = document.getElementById("loading");
const errorMessage = document.getElementById("error-message");
const resultsSection = document.getElementById("results");
const uploadSection = document.getElementById("upload-section");
const newAnalysisButton = document.getElementById("new-analysis");

let resumeFile = null;
let jobDescriptionFile = null;
let scoreAnimationFrame = null;


function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}


function clearError() {
    errorMessage.textContent = "";
    errorMessage.hidden = true;
}


function displaySelectedFile(element, file) {
    element.textContent = file ? `Selected: ${file.name}` : "";
}


function isPdf(file) {
    return file && file.name.toLowerCase().endsWith(".pdf");
}


function setFile(kind, file) {
    if (!isPdf(file)) {
        showError("Please upload a PDF file.");
        return;
    }

    if (kind === "resume") {
        resumeFile = file;
        displaySelectedFile(
            document.getElementById("resume-file"),
            file
        );
    } else {
        jobDescriptionFile = file;
        displaySelectedFile(
            document.getElementById("jd-file"),
            file
        );
    }

    clearError();
    analyzeButton.disabled = !(resumeFile && jobDescriptionFile);
}


function setupDropZone(dropZone, fileSetter) {
    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.style.borderColor = "rgba(255,255,255,0.4)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "";
    });

    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.style.borderColor = "";
        fileSetter(event.dataTransfer.files[0]);
    });
}


function renderTags(elementId, items, className = "") {
    const container = document.getElementById(elementId);
    container.replaceChildren();

    (items || []).forEach((item) => {
        const tag = document.createElement("span");
        tag.className = `tag ${className}`.trim();
        tag.textContent = item;
        container.appendChild(tag);
    });
}


function renderList(elementId, items) {
    const list = document.getElementById(elementId);
    list.replaceChildren();

    (items || []).forEach((item) => {
        const entry = document.createElement("li");
        entry.textContent = item;
        list.appendChild(entry);
    });
}


function setScoreVisual(score) {
    const normalizedScore = Math.max(0, Math.min(100, Number(score) || 0));
    const scoreElement = document.getElementById("score");
    const ringScoreElement = document.getElementById("ring-score");
    const scoreRing = document.querySelector(".score-ring");

    scoreElement.textContent = Math.round(normalizedScore);
    ringScoreElement.textContent = Math.round(normalizedScore);
    scoreRing.style.setProperty("--score-progress", `${normalizedScore}%`);
}


function animateScore(score) {
    const targetScore = Math.max(0, Math.min(100, Number(score) || 0));
    const animationDuration = 1400;
    const animationStart = performance.now();

    if (scoreAnimationFrame !== null) {
        cancelAnimationFrame(scoreAnimationFrame);
    }

    setScoreVisual(0);

    function updateScore(currentTime) {
        const elapsed = currentTime - animationStart;
        const progress = Math.min(elapsed / animationDuration, 1);
        const easedProgress = 1 - Math.pow(1 - progress, 3);

        setScoreVisual(targetScore * easedProgress);

        if (progress < 1) {
            scoreAnimationFrame = requestAnimationFrame(updateScore);
        } else {
            scoreAnimationFrame = null;
            setScoreVisual(targetScore);
        }
    }

    scoreAnimationFrame = requestAnimationFrame(updateScore);
}


function renderResults(result) {
    const analysis = result.analysis;

    animateScore(result.overall_score);
    document.getElementById("match-level").textContent = result.match_level;

    renderTags("matching-skills", analysis.matching_skills, "match");
    renderTags("missing-skills", analysis.missing_skills, "missing");
    renderTags("matching-keywords", analysis.matching_keywords, "match");
    renderTags("missing-keywords", analysis.missing_keywords, "missing");
    document.getElementById("experience-match").textContent =
        analysis.experience_match || "No experience assessment was returned.";
    renderList("strengths", analysis.strengths);
    renderList("recommendations", analysis.recommendations);
}


async function analyzeResume() {
    if (!resumeFile || !jobDescriptionFile) {
        showError("Upload both a resume and a job description PDF first.");
        return;
    }

    clearError();
    setScoreVisual(0);
    analyzeButton.disabled = true;
    uploadSection.hidden = true;
    analyzeButton.hidden = true;
    loadingSection.hidden = false;
    resultsSection.hidden = true;

    const formData = new FormData();
    formData.append("resume", resumeFile);
    formData.append("job_description", jobDescriptionFile);

    try {
        const response = await fetch("/ats/analyze", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "ATS analysis failed.");
        }

        renderResults(data);
        loadingSection.hidden = true;
        resultsSection.hidden = false;
    } catch (error) {
        loadingSection.hidden = true;
        uploadSection.hidden = false;
        analyzeButton.hidden = false;
        analyzeButton.disabled = false;
        showError(error.message || "Unable to analyze the uploaded files.");
    }
}


function resetAnalysis() {
    if (scoreAnimationFrame !== null) {
        cancelAnimationFrame(scoreAnimationFrame);
        scoreAnimationFrame = null;
    }

    resumeFile = null;
    jobDescriptionFile = null;
    resumeInput.value = "";
    jobDescriptionInput.value = "";
    document.getElementById("resume-file").textContent = "";
    document.getElementById("jd-file").textContent = "";
    resultsSection.hidden = true;
    loadingSection.hidden = true;
    uploadSection.hidden = false;
    analyzeButton.hidden = false;
    analyzeButton.disabled = true;
    setScoreVisual(0);
    clearError();
}


resumeInput.addEventListener("change", () => setFile("resume", resumeInput.files[0]));
jobDescriptionInput.addEventListener("change", () => setFile("job-description", jobDescriptionInput.files[0]));
analyzeButton.addEventListener("click", analyzeResume);
newAnalysisButton.addEventListener("click", resetAnalysis);

setupDropZone(resumeDropZone, (file) => setFile("resume", file));
setupDropZone(jobDescriptionDropZone, (file) => setFile("job-description", file));

// MCP Sample ATS Button Handler
const sampleAtsBtn = document.getElementById("sample-ats-btn");
if (sampleAtsBtn) {
    sampleAtsBtn.addEventListener("click", async () => {
        clearError();
        setScoreVisual(0);
        uploadSection.hidden = true;
        analyzeButton.hidden = true;
        sampleAtsBtn.hidden = true;
        loadingSection.hidden = false;
        resultsSection.hidden = true;

        try {
            const response = await fetch("/ats/sample", { method: "POST" });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || "Sample ATS analysis failed.");
            }
            renderResults(data);
            loadingSection.hidden = true;
            resultsSection.hidden = false;
        } catch (error) {
            loadingSection.hidden = true;
            uploadSection.hidden = false;
            analyzeButton.hidden = false;
            sampleAtsBtn.hidden = false;
            analyzeButton.disabled = !(resumeFile && jobDescriptionFile);
            showError(error.message || "Failed to run sample ATS analysis.");
        }
    });
}
