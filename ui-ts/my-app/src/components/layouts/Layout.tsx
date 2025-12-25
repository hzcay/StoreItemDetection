import { ReactNode } from "react"
import { Card } from "@/components/ui/card"
import { Header } from "./Header"
import { Sidebar } from "./SideBar"
import { Footer } from "./Footer"

// ================= Layout =================
export default function Layout({ children }: { children: ReactNode }) {
    return (
        <div className="h-screen flex flex-col">
            <Header />
            <div className="flex flex-1 overflow-hidden">
                <Sidebar />
                <main className="flex-1 p-6 overflow-y-auto">
                    <Card className="p-6">{children}</Card>
                </main>
            </div>
            <Footer />
        </div>
    )
}
