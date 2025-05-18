import { NextResponse } from "next/server"
import path from "path"
import fs from "fs"
import csv from "csv-parser"

export async function GET() {
  const results: any[] = []
    const filePath = path.join(process.cwd(), "..", "data processing", "estimated_times.csv");

  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(csv())
      .on("data", (data) => results.push(data))
      .on("end", () => resolve(NextResponse.json(results)))
      .on("error", (err) =>
        resolve(NextResponse.json({ error: err.message }, { status: 500 }))
      )
  })
}