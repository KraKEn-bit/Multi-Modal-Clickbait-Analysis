import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { LenisProvider } from "@/components/LenisProvider";
import "./globals.css";
import "lenis/dist/lenis.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

export const metadata: Metadata = {
  title: "VTCF Agent — Bangla Clickbait Detector",
  description:
    "An autonomous visual-temporal agent that watches Bangla YouTube videos and calls out clickbait before you press play.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrains.variable} h-full`}>
      <body className="min-h-full antialiased">
        <LenisProvider>{children}</LenisProvider>
      </body>
    </html>
  );
}
