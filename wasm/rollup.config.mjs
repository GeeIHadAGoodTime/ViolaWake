import resolve from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";

export default [
  // ESM build (for bundlers / modern browsers via <script type="module">)
  {
    input: "src/index.ts",
    output: {
      file: "dist/violawake.js",
      format: "es",
      sourcemap: true,
    },
    plugins: [
      resolve({ browser: true }),
      typescript({ tsconfig: "./tsconfig.json", declaration: true, declarationDir: "dist" }),
    ],
    // onnxruntime-web must be loaded separately via CDN in browser demo
    // (it ships its own WASM assets that need to live next to the bundle).
    // In a bundler environment (webpack/vite) the user controls this.
    external: ["onnxruntime-web"],
  },
  // CJS build (for Node / Jest)
  {
    input: "src/index.ts",
    output: {
      file: "dist/violawake.cjs",
      format: "cjs",
      sourcemap: true,
    },
    plugins: [
      resolve({ browser: false }),
      typescript({ tsconfig: "./tsconfig.json", declaration: false }),
    ],
    external: ["onnxruntime-web"],
  },
];
