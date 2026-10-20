import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "电商客服 Agent",
  description: "商品咨询、订单查询与售后知识助手",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
