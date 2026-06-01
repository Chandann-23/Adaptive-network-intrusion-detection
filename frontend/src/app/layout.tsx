import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Adaptive NIDS — Zero-Day Network Intrusion Detection",
  description:
    "Production-grade Network Intrusion Detection System. Two-stage hybrid pipeline combining Isolation Forest novelty detection and XGBoost multiclass classification. 89.4% novel attack recall on NSL-KDD.",
  keywords: ["NIDS", "intrusion detection", "machine learning", "cybersecurity", "XGBoost", "zero-day"],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
