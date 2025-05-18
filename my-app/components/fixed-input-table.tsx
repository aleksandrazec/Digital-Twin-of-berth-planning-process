"use client"
import { useEffect, useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface Ship {
  CALL_SIGN: string
  VESSEL_NAME: string
  AGENT_NAME: string
  ETA_TIME: string
  ETD_TIME: string
}

export default function FixedInputTable({data}: {data: any[]}) {
  const [ships, setShips] = useState<Ship[]>([])
  useEffect(() => {
    fetch("/api/ships")
      .then(res => res.json())
      .then(data => setShips(data))
  }, [])
  return (
    <div className="overflow-x-auto p-4">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Call Sign</TableHead>
            <TableHead>ETA</TableHead>
            <TableHead>ETD</TableHead>
            <TableHead>Vessel Name</TableHead>
            <TableHead>Agent</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {ships.map((ship, index) => (
            <TableRow key={index}>
              <TableCell>{ship.CALL_SIGN}</TableCell>
              <TableCell>{ship.ETA_TIME}</TableCell>
              <TableCell>{ship.ETD_TIME}</TableCell>
              <TableCell>{ship.VESSEL_NAME}</TableCell>
              <TableCell>{ship.AGENT_NAME}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}