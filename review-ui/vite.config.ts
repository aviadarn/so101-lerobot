import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base so the built page works from a plain file server, a subpath, or an
// artifact host without knowing its mount point in advance.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", assetsInlineLimit: 0 },
});
