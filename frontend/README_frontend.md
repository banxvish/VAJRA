# KAVACHA AI Dashboard (Frontend)

The Frontend represents the visual analytics dashboard and central operations command center for the KAVACHA AI Voice Defense Engine. It's built with modern web technologies focused on creating a sleek, highly responsive, and dynamic UI for evaluating live audio traces.

## Setup and Installation

Before you begin, ensure you have Node.js (v18+) and npm installed.

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install all node module dependencies
npm install
```

## Running the React Dashboard

To launch the dashboard for local development and real-time inference monitoring:
```bash
npm run dev
```

The web application will instantly launch on `http://localhost:5173`. 
Note: Ensure the backend instance is up and running simultaneously at port `8000` to make successful API calls.

## API Integration Instructions

The frontend communicates with the FastAPI layer securely, primarily wrapping standard `FormData` POST API transactions.

- Upload files are captured using HTML5 `<input type="file" />` elements and mapped onto `FormData`. 
- The React application sends a `POST` request to `http://127.0.0.1:8000/analyze_audio`, which then synchronously awaits the deep learning responses.
- The UI gracefully blocks and renders a loading state animation to manage the `< 2s` inference latency properly.
- Results derived from the JSON Response payload (`trust_score`, individual model indicators, and security status) are mapped to visual components like progress bars and warning notifications.

## UI Architecture

The KAVACHA UI integrates multiple high-end developer libraries:
- **React 18**: Underlying state and core component tree rendering.
- **Vite**: Rapid-fast HMR and bundle compilation.
- **TypeScript**: Static typing for error-proof API data definitions.
- **TailwindCSS**: Inline utility parsing mapping visual tokens dynamically.
- **shadcn-ui**: Premium accessible React primitives used for buttons, interactive cards, progress dials, and toast notifications.
