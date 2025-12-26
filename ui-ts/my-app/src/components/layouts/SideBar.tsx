import { Button } from "@/components/ui/button";
import { useNavigate, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

// Navigation items with their paths
const navItems = [
    { name: "Dashboard", path: "/dashboard" },
    { name: "Products", path: "/products" },
    { name: "Categories", path: "/categories" },
    { name: "Orders", path: "/orders" },
    { name: "Display", path: "/home" },
];

export function Sidebar() {
    const navigate = useNavigate();
    const location = useLocation();
    const [activePath, setActivePath] = useState(location.pathname);

    // Update active path when location changes
    useEffect(() => {
        setActivePath(location.pathname);
    }, [location]);

    const handleNavigation = (path: string) => {
        // Add a small delay for a smoother transition
        setActivePath(path);
        setTimeout(() => {
            navigate(path);
        }, 150);
    };

    return (
        <aside className="w-64 border-r h-full p-4 bg-muted/30 transition-all duration-200">
            <nav className="space-y-2">
                {navItems.map((item) => {
                    const isActive = activePath.startsWith(item.path);
                    return (
                        <Button
                            key={item.path}
                            variant={isActive ? "secondary" : "ghost"}
                            className={cn(
                                "w-full justify-start transition-all duration-200 transform hover:translate-x-1",
                                isActive && "font-semibold"
                            )}
                            onClick={() => handleNavigation(item.path)}
                        >
                            {item.name}
                        </Button>
                    );
                })}
            </nav>
        </aside>
    );
}