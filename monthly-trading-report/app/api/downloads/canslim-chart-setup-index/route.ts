import { readFile } from "fs/promises";
import path from "path";

export async function GET() {
  const filePath = path.join(process.cwd(), "public", "canslim-chart-setup-index.csv");
  const file = await readFile(filePath);

  return new Response(file, {
    headers: {
      "Content-Disposition": 'attachment; filename="canslim-chart-setup-index.csv"',
      "Content-Type": "text/csv; charset=utf-8",
      "Cache-Control": "public, max-age=300"
    }
  });
}
