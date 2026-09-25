// Shared section container + header (layout pass, 25 Sep 2026).
// Every metrics / findings block in Chewsy uses the SAME recipe: a tinted
// well (12px radius, 16px padding, tonal layer — never a nested card)
// headed by title-left / muted-caption-right. This is the single source
// of that pattern; do not inline variants.

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Well({
  children,
  className,
  as: Tag = "div",
  ...rest
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "section";
} & Omit<React.ComponentProps<"div">, "children">) {
  return (
    <Tag className={cn("rounded-lg bg-muted/60 p-4", className)} {...rest}>
      {children}
    </Tag>
  );
}

export function SectionHead({ title, caption }: { title: string; caption?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <h2 className="text-sm font-semibold">{title}</h2>
      {caption && <p className="text-xs text-muted-foreground">{caption}</p>}
    </div>
  );
}
