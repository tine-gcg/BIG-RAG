import { Message } from "../lib/types";

export default function ChatBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div
      className={`p-3 max-w-[80%] rounded-xl ${
        isUser
          ? "bg-blue-600 text-white ml-auto"
          : "bg-gray-200 text-black mr-auto"
      }`}
    >
      {message.content}
    </div>
  );
}
