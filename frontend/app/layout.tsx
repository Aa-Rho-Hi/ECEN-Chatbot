import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TAMU ECE Assistant",
  description: "RAG-powered chatbot for the TAMU Electrical & Computer Engineering department",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
