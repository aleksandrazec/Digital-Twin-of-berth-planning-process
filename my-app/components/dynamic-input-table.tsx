"use client"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Slider } from "@/components/ui/slider"

interface DynamicInputTableProps {
  data: Record<
    number,
    {
      Effectiveness: number
      Reliability: number
      Work_Environment: number
      Weather: number
      Congestion: number
    }
  >
  shipIds: number[]
  onChange: (shipId: number, field: string, value: number) => void
}

export function DynamicInputTable({ data, shipIds, onChange }: DynamicInputTableProps) {
  const fields = [
    { name: "Effectiveness", label: "Effectiveness" },
    { name: "Reliability", label: "Reliability" },
    { name: "Work_Environment", label: "Work Environment" },
    { name: "Weather", label: "Weather" },
    { name: "Congestion", label: "Congestion" },
  ]

  const handleSliderChange = (shipId: number, field: string, value: number[]) => {
    onChange(shipId, field, value[0])
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Ship ID</TableHead>
            {fields.map((field) => (
              <TableHead key={field.name}>{field.label}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {shipIds.map((shipId) => (
            <TableRow key={shipId}>
              <TableCell>{shipId}</TableCell>
              {fields.map((field) => (
                <TableCell key={`${shipId}-${field.name}`} className="min-w-[150px]">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">
                        {data[shipId][field.name as keyof (typeof data)[typeof shipId]].toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      value={[data[shipId][field.name as keyof (typeof data)[typeof shipId]]]}
                      min={0}
                      max={1}
                      step={0.01}
                      onValueChange={(value) => handleSliderChange(shipId, field.name, value)}
                    />
                  </div>
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
