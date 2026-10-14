"use client";
import { useEffect, useRef } from "react";
import { init, use, type EChartsCoreOption, type ECharts } from "echarts/core";
import { LineChart, BarChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkAreaComponent,
  AriaComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
use([
  LineChart,
  BarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkAreaComponent,
  AriaComponent,
  CanvasRenderer,
]);

export function Chart({
  option,
  label,
  height = 300,
}: {
  option: EChartsCoreOption;
  label: string;
  height?: number;
}) {
  const element = useRef<HTMLDivElement>(null);
  const instance = useRef<ECharts | null>(null);
  useEffect(() => {
    if (!element.current) return;
    instance.current = init(element.current);
    const resize = new ResizeObserver(() => instance.current?.resize());
    resize.observe(element.current);
    return () => {
      resize.disconnect();
      instance.current?.dispose();
      instance.current = null;
    };
  }, []);
  useEffect(() => {
    instance.current?.setOption(
      {
        animation: false,
        color: ["#297f70", "#293d4e", "#b5c9c4", "#d6a461"],
        textStyle: { fontFamily: "Arial, sans-serif", color: "#68777d" },
        grid: { left: 65, right: 25, top: 35, bottom: 50 },
        tooltip: { trigger: "axis", renderMode: "richText" },
        aria: { enabled: true, description: label },
        ...option,
      },
      true,
    );
  }, [option, label]);
  return (
    <div
      className="chart"
      role="img"
      aria-label={label}
      ref={element}
      style={{ height }}
    />
  );
}

export const axis = {
  axisLine: { show: false },
  axisTick: { show: false },
  axisLabel: { color: "#79868b", fontSize: 11 },
  splitLine: { lineStyle: { color: "#edf0ed", type: "dashed" } },
};
