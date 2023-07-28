from dataclasses import dataclass

@dataclass
class Item:
    name: str
    unit_price: float
    quantity: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity
    
if __name__ == "__main__":
    
    pineapple = Item(name="pineapple", unit_price=5, quantity=10)                                  

    