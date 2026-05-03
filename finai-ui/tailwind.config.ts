import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-dm-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "monospace"],
      },
      colors: {
        surface: {
          DEFAULT: "#0a0e17",
          secondary: "#111827",
          card: "#1a2235",
          hover: "#1f2a40",
          input: "#0d1321",
        },
        border: {
          DEFAULT: "#2a3550",
          focus: "#3b82f6",
        },
        txt: {
          primary: "#e8edf5",
          secondary: "#8896b0",
          muted: "#5a6a85",
        },
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
      },
      keyframes: {
        fadeIn: {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        slideUp: {
          from: { opacity: "0", transform: "translateY(12px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
