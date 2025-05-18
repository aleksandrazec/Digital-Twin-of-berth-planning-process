"use client"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Slider } from "@/components/ui/slider"

interface DynamicInputTableProps {
  data: Record<
    string,
    {
      Effectiveness: number
      Reliability: number
      Work_Environment: number
      Weather: number
      Congestion: number
    }
  >
  shipIds: string[]
  onChange: (shipId: string, field: string, value: number) => void
}

export function DynamicInputTable({ data, shipIds, onChange }: DynamicInputTableProps) {
  const fields = [
    { name: "Effectiveness", label: "Effectiveness" },
    { name: "Reliability", label: "Reliability" },
    { name: "Work_Environment", label: "Work Environment" },
    { name: "Weather", label: "Weather" },
    { name: "Congestion", label: "Congestion" },
  ]

  const handleSliderChange = (shipId: string, field: string, value: number[]) => {
    onChange(shipId, field, value[0])
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            {/* <TableHead>Call Sign</TableHead> */}
            {fields.map((field) => (
              <TableHead key={field.name}>{field.label}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {shipIds.map((shipId) => (
            <TableRow key={shipId}>
              {/* <TableCell>{shipId}</TableCell> */}
              {fields.map((field) => {
                const value = data[shipId]?.[field.name as keyof typeof data[string]]
                return (
                  <TableCell key={`${shipId}-${field.name}`} className="min-w-[150px]">
                    {value !== undefined ? (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm">{typeof value === "number" ? value.toFixed(2) : Number(value).toFixed(2)}</span>
                        </div>
                        <Slider
                          value={[value]}
                          min={0}
                          max={1}
                          step={0.01}
                          onValueChange={(val) => handleSliderChange(shipId, field.name, val)}
                        />
                      </div>
                    ) : (
                      <span className="text-muted">Loading...</span>
                    )}
                  </TableCell>
                )
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
