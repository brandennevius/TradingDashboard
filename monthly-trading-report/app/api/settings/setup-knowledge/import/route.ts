import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";

const MAX_FILE_BYTES = 12 * 1024 * 1024;
const MAX_EXTRACTED_CHARS = 120_000;
const TARGET_CHUNK_CHARS = 3_500;
const MAX_CHUNK_CHARS = 5_500;

function cleanText(value: string) {
  return value
    .replace(/\r/g, "\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim()
    .slice(0, MAX_EXTRACTED_CHARS);
}

function chunkTitle(chunk: string, index: number) {
  const firstLine = chunk.split("\n").map((line) => line.trim()).find(Boolean) || "";
  const cleaned = firstLine.replace(/^#+\s*/, "").replace(/[:.]\s*$/, "").slice(0, 90);
  return cleaned.length >= 8 ? cleaned : `Section ${index + 1}`;
}

function splitLongSection(section: string) {
  if (section.length <= MAX_CHUNK_CHARS) return [section];

  const paragraphs = section.split(/\n{2,}/).map((paragraph) => paragraph.trim()).filter(Boolean);
  const chunks: string[] = [];
  let current = "";

  for (const paragraph of paragraphs) {
    if (current && `${current}\n\n${paragraph}`.length > TARGET_CHUNK_CHARS) {
      chunks.push(current.trim());
      current = paragraph;
    } else {
      current = current ? `${current}\n\n${paragraph}` : paragraph;
    }
  }

  if (current.trim()) chunks.push(current.trim());

  return chunks.flatMap((chunk) => {
    if (chunk.length <= MAX_CHUNK_CHARS) return [chunk];
    const parts: string[] = [];
    for (let start = 0; start < chunk.length; start += TARGET_CHUNK_CHARS) {
      parts.push(chunk.slice(start, start + TARGET_CHUNK_CHARS).trim());
    }
    return parts.filter(Boolean);
  });
}

function chunkStrategyText(text: string) {
  const cleaned = cleanText(text);
  const roughSections = cleaned
    .split(/\n(?=(?:#{1,4}\s+|[A-Z][A-Z0-9 /&().,'-]{8,}$|\d+\.\s+[A-Z]))/gm)
    .map((section) => section.trim())
    .filter(Boolean);
  const sections = roughSections.length ? roughSections : [cleaned];
  const chunks = sections.flatMap(splitLongSection).filter((section) => section.length >= 80);

  return chunks.map((content, index) => ({
    id: `chunk-${index + 1}`,
    title: chunkTitle(content, index),
    content,
    order: index
  }));
}

async function extractPdf(buffer: Buffer) {
  const { PDFParse } = await import("pdf-parse");
  const parser = new PDFParse({ data: buffer });

  try {
    const result = await parser.getText();
    return cleanText(result.text || "");
  } finally {
    await parser.destroy();
  }
}

async function extractDocx(buffer: Buffer) {
  const mammoth = await import("mammoth");
  const result = await mammoth.extractRawText({ buffer });
  return cleanText(result.value || "");
}

async function extractText(file: File, buffer: Buffer) {
  const lowerName = file.name.toLowerCase();
  const mimeType = file.type.toLowerCase();

  if (mimeType.includes("pdf") || lowerName.endsWith(".pdf")) {
    return extractPdf(buffer);
  }

  if (
    mimeType.includes("wordprocessingml") ||
    mimeType.includes("msword") ||
    lowerName.endsWith(".docx")
  ) {
    return extractDocx(buffer);
  }

  if (mimeType.startsWith("text/") || lowerName.endsWith(".txt") || lowerName.endsWith(".md")) {
    return cleanText(buffer.toString("utf8"));
  }

  throw new Error("Unsupported strategy document type. Upload a PDF, DOCX, TXT, or Markdown file.");
}

export async function POST(request: Request) {
  const user = await getSessionUser();

  if (!user || user.id !== "branden" || user.readOnly) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  const formData = await request.formData();
  const file = formData.get("file");

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "Upload a strategy document file." }, { status: 400 });
  }

  if (file.size > MAX_FILE_BYTES) {
    return NextResponse.json({ error: "Strategy document is too large. Keep uploads under 12MB." }, { status: 400 });
  }

  try {
    const buffer = Buffer.from(await file.arrayBuffer());
    const content = await extractText(file, buffer);

    if (!content) {
      return NextResponse.json({ error: "Could not extract readable text from that document." }, { status: 422 });
    }

    return NextResponse.json({
      title: file.name.replace(/\.[^.]+$/, ""),
      sourceType: "document",
      url: file.name,
      content,
      chunks: chunkStrategyText(content),
      truncated: content.length >= MAX_EXTRACTED_CHARS
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not import the strategy document." },
      { status: 400 }
    );
  }
}
