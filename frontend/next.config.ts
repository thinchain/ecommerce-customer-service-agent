import type { NextConfig } from "next";

const backendUrl = process.env.FASTAPI_BASE_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: { proxyTimeout: Number(process.env.AGENT_PROXY_TIMEOUT_MS || 630000) },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
