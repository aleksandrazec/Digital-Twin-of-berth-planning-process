export interface Ship {
  id: number
  Type: number
  Size: number
  Draft: number
  ETA: number
  ETD: number
  Priority_Of_Shipment: number
}

export interface AllocationResult {
  shipId: number
  ATA: number
  ATD: number
  Berth_No: number
}
