import type { FieldDelta } from "../types";

interface Props {
  deltas: FieldDelta[];
  title: string;
}

function fmt(v: number | null): string {
  if (v == null) return "--";
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

export default function DeltaTable({ deltas, title }: Props) {
  if (!deltas.length) return null;
  return (
    <div className="delta-table-wrapper">
      <div className="delta-table-title">{title}</div>
      <table className="delta-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Source</th>
            <th>Output</th>
            <th>Target</th>
            <th>Delta</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {deltas.map((d) => (
            <tr key={d.field} className={`status-${d.status}`}>
              <td className="field-name">{d.field}</td>
              <td>{fmt(d.src_val)}</td>
              <td>{fmt(d.out_val)}</td>
              <td>{fmt(d.tgt_val)}</td>
              <td>{fmt(d.delta_src_out)}</td>
              <td>
                <span className={`status-badge ${d.status}`}>{d.status}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
