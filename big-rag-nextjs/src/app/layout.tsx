import "./globals.css";

export const metadata = {
  title: "AI Chatbot",
  description: "Next.js AI chatbot with voice replies",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-800">{children}</body>
    </html>
  );
}
