import type { Metadata } from "next";
import { Workspace } from "@/components/workspace";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "DecisionGuard · Marketing intelligence",
    template: "%s · DecisionGuard",
  },
  description:
    "Evidence-led marketing decisions. Model uncertainty, validation and budget scenarios in one workspace.",
  robots: { index: false, follow: false },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Workspace>{children}</Workspace>
      </body>
    </html>
  );
}
