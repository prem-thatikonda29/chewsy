import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Product images come from Open Food Facts' CDN (image_url in the
  // /predict response). Next 16: images.domains is deprecated — use
  // remotePatterns (verified against node_modules/next/dist/docs upgrade
  // guide). Wildcard keeps rotating image sizes/cache-buster query
  // variants working.
  images: {
    remotePatterns: [
      new URL("https://images.openfoodfacts.org/**"),
      new URL("https://static.openfoodfacts.org/**"),
    ],
  },
};

export default nextConfig;
