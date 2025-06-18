import type { NextApiRequest, NextApiResponse } from "next";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";

export const config = { api: { bodyParser: { sizeLimit: "5mb" } } };

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });
  const { text } = req.body;
  const id = randomUUID();
  const outputPath = path.join("/tmp", `${id}.wav`);

  try {
    await new Promise((resolve, reject) => {
      const child = spawn("python3", [
        "scripts/tts_kokoro.py",
        text,
        outputPath,
      ]);
      child.stderr.on("data", (d) => console.error(d.toString()));
      child.on("exit", (code) =>
        code === 0 ? resolve(null) : reject(new Error(`Exit ${code}`))
      );
    });

    const buf = fs.readFileSync(outputPath);
    fs.unlinkSync(outputPath);
    res.status(200).json({ audioBase64: buf.toString("base64") });
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: e.message || "TTS failed" });
  }
}
