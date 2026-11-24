# Final Application Validation: Grounded Side-Burnt Document Text Restoration

**Project:** Grounded Side-Burnt Document Text Restoration  
**Date:** November 2026  
**Document Status:** AUTHORITATIVE BACKEND & FRONTEND VALIDATION REPORT  
**Application Entrypoint:** `api/main.py`  
**Workbench URL:** `http://127.0.0.1:8000/`  

---

## 1. Backend REST API Endpoint Verification

All REST API endpoints were validated via automated unit testing (`tests/test_api.py`) and live HTTP transaction testing (`scripts/test_frontend_backend_flow.py`):

| HTTP Method | Route | Request Payload | Response Status | Verified Functionality |
|---|---|---|---|---|
| `GET` | `/health` | None | **200 OK** | Server status (`healthy`), device (`cpu`), model readiness (`true`). |
| `GET` | `/model/info` | None | **200 OK** | Architecture (`ForensicT5Restorer`), total parameters (**85,476,929**). |
| `GET` | `/` | None | **200 OK** | Serves interactive research workbench single-page application. |
| `POST` | `/restore/image` | `multipart/form-data` (file, max_lines, beams) | **200 OK** | Auto burn detection, multi-line neural restoration, Base64 overlay. |
| `POST` | `/restore/page` | `multipart/form-data` (alias route) | **200 OK** | Verified seamless alias mapping to `/restore/image`. |
| `POST` | `/restore/region` | `multipart/form-data` (file, x1, y1, x2, y2, prefix) | **200 OK** | Single severed line neural infilling with physical budget report. |
| `POST` | `/restore/batch` | `multipart/form-data` (multiple files) | **200 OK** | Sequential batch processing with individual document status. |

---

## 2. API Edge-Case & Error Handling Audit

Defensive input validation and HTTP exception handling were audited across failure modes:

| Test Case | Injected Fault | Expected Status | Observed Behavior |
|---|---|---|---|
| **Empty File Upload** | `file = b""` (0 bytes) | `400 Bad Request` | [x] PASSED (`"Uploaded file is empty"`) |
| **Corrupt / Non-Image File** | `text/plain` file content | `400 Bad Request` | [x] PASSED (`"Failed to decode uploaded image"`) |
| **Malformed Bounding Box** | `manual_bbox = "invalid,bbox"` | `400 Bad Request` | [x] PASSED (`"Must provide 4 integers"`) |
| **Batch Size Exceeded** | 12 files uploaded ($> 10$) | `400 Bad Request` | [x] PASSED (`"Batch size exceeds maximum limit of 10"`) |
| **Uncaught Server Exception** | Runtime error | `500 Server Error` | [x] PASSED (Structured JSON error envelope) |

---

## 3. Interactive Web Workbench Frontend Audit (`api/static/index.html`)

The interactive frontend is implemented as a lightweight, dependency-free HTML5 application powered by Tailwind CSS and HTML5 Canvas.

### A. Implemented UI Capabilities:
1. **Live Health Polling:** Automatically queries `/health` on load and updates the navigation bar status badge (`Server Ready (CPU)`).
2. **Drag-and-Drop File Upload:** Intuitive `#drop-zone` supporting PNG, JPG, and TIFF image formats.
3. **Dual Mode Selection:**
   - **Auto Detection (OpenCV):** Automatically runs horizontal projection and charcoal contour extraction.
   - **Manual Region Selection:** Displays canvas crosshair tool allowing users to click and drag a custom bounding box around the damaged margin.
4. **Interactive Document Canvas:** High-resolution rendering on `#doc-canvas` with coordinate scaling.
5. **View Toggle:** Seamlessly switches between the original image and the visual restoration overlay with severed line indicators.
6. **Performance Telemetry:** Displays total page latency, severed line count, and average latency per line.
7. **Line-by-Line Restorations Card:**
   - Surviving visible prefix text
   - Restored missing text hypothesis
   - Physical character budget and margin violation badge
   - Damage margin pixel width and inference latency
8. **Export Tools:**
   - **Copy Text:** Formats and copies all reconstructed lines to the system clipboard.
   - **Export JSON:** Generates and downloads a machine-readable JSON restoration report.

---

## 4. End-to-End Simulation & Browser Verification

### Automated Workflow Simulation:
The simulation script `scripts/test_frontend_backend_flow.py` executed against the live server:
- Verified static HTML workbench delivery (19,766 bytes).
- Uploaded `data/synthetic_fixture/page_001.png` via multipart POST.
- Restored 2 test lines in $3,924.5\text{ ms}$.
- Verified Base64 diagnostic overlay decoding ($181,532\text{ bytes}$).
- Tested manual bounding box selection override (`is_manual: True`).
- Validated rejection of corrupt file inputs.

### Browser Automation Status:
Chrome remote debugging was not active on the host machine (`DevToolsActivePort` was not present in the user profile directory). Consequently, automated browser click-through via the DevTools protocol could not attach directly. All underlying DOM element IDs, canvas event listeners, and API network calls were validated through code audit and automated HTTP transaction tests.

### Manual Verification Guide for Live College Demonstration:
1. Start the server:
   ```bash
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. Open Chrome, Edge, or Firefox to `http://127.0.0.1:8000/`.
3. Drag `data/synthetic_fixture/page_001.png` onto the drop zone.
4. Verify the document appears on the canvas.
5. Click **Execute Restoration**.
6. Observe line restorations appearing in the right column and click **Export JSON**.

