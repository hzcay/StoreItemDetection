
import { Button } from "@/components/ui/button"
import { Menu } from "lucide-react"

// ================= Header =================
export function Header() {
    return (
        <header className="h-14 border-b flex items-center justify-between px-6 bg-background">
            <div className="flex items-center gap-2">
                <Menu className="h-5 w-5" />
                <span className="font-semibold text-lg">Store Admin</span>
            </div>
            <Button size="sm">Logout</Button>
        </header>
    )
}

