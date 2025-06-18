import type { NextApiRequest, NextApiResponse } from "next";

const WEBHOOK = process.env.NEXT_PUBLIC_WEBHOOK_URL!;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  const { chatInput, sessionId, accessToken } = req.body;
  try {
    const resp = await fetch(WEBHOOK, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ chatInput, sessionId }),
    });
    if (!resp.ok)
      return res.status(resp.status).json({ error: await resp.text() });
    const data = await resp.json();
    res.status(200).json({ output: data.output });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
}
