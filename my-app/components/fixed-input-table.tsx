import type { Ship } from "@/lib/types"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface FixedInputTableProps {
  data: Ship[]
}

export function FixedInputTable({ data }: FixedInputTableProps) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Ship ID</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Size</TableHead>
            <TableHead>Draft</TableHead>
            <TableHead>ETA</TableHead>
            <TableHead>ETD</TableHead>
            <TableHead>Priority</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((ship) => (
            <TableRow key={ship.id}>
              <TableCell>{ship.id}</TableCell>
              <TableCell>{ship.Type}</TableCell>
              <TableCell>{ship.Size}</TableCell>
              <TableCell>{ship.Draft}</TableCell>
              <TableCell>{ship.ETA}</TableCell>
              <TableCell>{ship.ETD}</TableCell>
              <TableCell>{ship.Priority_Of_Shipment}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
