// Workbook writer for prepare_jmp_retention_data_vbg0.py.
// All scientific calculations and quality checks are in the Python companion.
// This writer preserves typed numbers and a single import-friendly header row.

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputPath] = process.argv.slice(2);
if (!payloadPath || !outputPath) {
  throw new Error("Usage: node prepare_jmp_retention_data_vbg0.mjs payload.json output.xlsx");
}
const sheets = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const expected = ["All_Data", "PostBaseline", "CheckpointSummary", "ColumnInfo"];
if (JSON.stringify(Object.keys(sheets)) !== JSON.stringify(expected)) {
  throw new Error("Unexpected sheet order or names in the preparation payload");
}

const workbook = Workbook.create();
const font = "Arial";
const headerFill = "#25476D";
const textColor = "#192532";

function applyWidths(sheet, widths) {
  widths.forEach((width, columnIndex) => {
    sheet.getCell(0, columnIndex).format.columnWidth = width;
  });
}

for (const name of expected) {
  const rows = sheets[name];
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  if (!Array.isArray(rows) || rows.length < 2 || !Array.isArray(rows[0])) {
    throw new Error(`Empty or malformed worksheet payload: ${name}`);
  }
  const cols = rows[0].length;
  if (rows.some((row) => !Array.isArray(row) || row.length !== cols)) {
    throw new Error(`Uneven row widths in ${name}`);
  }
  sheet.getRangeByIndexes(0, 0, rows.length, cols).values = rows;
  const used = sheet.getRangeByIndexes(0, 0, rows.length, cols);
  used.format.font = { name: font, size: 10, color: textColor };
  used.format.verticalAlignment = "center";
  const header = sheet.getRangeByIndexes(0, 0, 1, cols);
  header.format.fill = headerFill;
  header.format.font = { name: font, size: 10, bold: true, color: "#FFFFFF" };
  header.format.rowHeight = 28;
  header.format.horizontalAlignment = "center";
  sheet.freezePanes.freezeRows(1);

  if (name === "All_Data" || name === "PostBaseline") {
    applyWidths(sheet, [15, 22, 10, 10, 12, 23, 23, 23, 24, 22, 22, 25]);
    const n = rows.length - 1;
    for (const column of [2, 3, 4]) {
      sheet.getRangeByIndexes(1, column, n, 1).setNumberFormat("0");
    }
    for (const column of [5, 6, 7, 8, 11]) {
      sheet.getRangeByIndexes(1, column, n, 1).setNumberFormat("0.000000000000000");
    }
    for (const column of [9, 10]) {
      sheet.getRangeByIndexes(1, column, n, 1).setNumberFormat("0.000000000000");
    }
  } else if (name === "CheckpointSummary") {
    applyWidths(sheet, [12, 15, 28, 25, 28, 25]);
    sheet.getRangeByIndexes(1, 0, rows.length - 1, 2).setNumberFormat("0");
    sheet.getRangeByIndexes(1, 2, rows.length - 1, 4)
      .setNumberFormat("0.000000000000");
  } else {
    applyWidths(sheet, [24, 27, 68, 23, 104, 30]);
    sheet.getRangeByIndexes(1, 0, rows.length - 1, cols).format.rowHeight = 62;
    sheet.getRangeByIndexes(1, 2, rows.length - 1, 1).format.wrapText = true;
    sheet.getRangeByIndexes(1, 4, rows.length - 1, 1).format.wrapText = true;
    sheet.getRangeByIndexes(1, 0, rows.length - 1, cols).format.verticalAlignment = "center";
  }
}

workbook.recalculate();
for (const [sheetName, range] of [
  ["All_Data", "A1:L6"],
  ["PostBaseline", "A1:L6"],
  ["CheckpointSummary", "A1:F7"],
  ["ColumnInfo", "A1:F8"],
]) {
  const check = await workbook.inspect({
    kind: "table", range: `${sheetName}!${range}`,
    include: "values,formulas", tableMaxRows: 8, tableMaxCols: 12,
    maxChars: 1200,
  });
  console.log(`${sheetName} inspect: ${check.ndjson}`);
}
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 30 },
  summary: "final formula error scan",
  maxChars: 1500,
});
console.log(`Error scan: ${errors.ndjson}`);

// Render on demand for visual QA. The delivered workbook does not depend on previews.
if (process.env.JMP_RENDER_PREVIEWS === "1") {
  const previewDir = process.env.JMP_PREVIEW_DIR ||
    path.join(path.dirname(payloadPath), "previews");
  await fs.mkdir(previewDir, { recursive: true });
  for (const [sheetName, range] of [
    ["All_Data", "A1:L5"],
    ["PostBaseline", "A1:L5"],
    ["CheckpointSummary", "A1:F7"],
    ["ColumnInfo", "A1:F7"],
  ]) {
    const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
    await fs.writeFile(path.join(previewDir, `${sheetName}.png`),
                       new Uint8Array(await preview.arrayBuffer()));
  }
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(`Saved ${outputPath}`);
