"use client";
import { Chart, axis } from "./chart";
import type { Predictions } from "@/lib/types";

export function PredictionChart({
  data,
  residuals = false,
}: {
  data: Predictions;
  residuals?: boolean;
}) {
  const points = data.rows;
  const holdout = points.find((item) => item.split === "holdout");
  return (
    <Chart
      label={
        residuals
          ? "Prediction residuals by week, AUD"
          : "Actual and predicted weekly revenue with 90% posterior interval, AUD"
      }
      option={{
        legend: {
          top: 0,
          right: 10,
          itemWidth: 16,
          itemHeight: 8,
          textStyle: { fontSize: 11, color: "#6b777d" },
          data: residuals
            ? ["Residual"]
            : ["Actual revenue", "Predicted revenue", "90% interval"],
        },
        xAxis: {
          ...axis,
          type: "category",
          data: points.map((item) => item.week),
          axisLabel: {
            ...axis.axisLabel,
            formatter: (value: string) => value.slice(0, 7),
          },
        },
        yAxis: {
          ...axis,
          type: "value",
          axisLabel: {
            ...axis.axisLabel,
            formatter: (value: number) =>
              Math.abs(value) >= 1000 ? `${value / 1000}k` : String(value),
          },
        },
        series: residuals
          ? [
              {
                name: "Residual",
                type: "bar",
                data: points.map((item) => item.residual),
                barMaxWidth: 5,
              },
            ]
          : [
              {
                name: "90% interval",
                type: "line",
                data: points.map((item) => item.lower_90),
                symbol: "none",
                lineStyle: { color: "#a5bfb8", width: 1, type: "dashed" },
                tooltip: { show: false },
              },
              {
                name: "90% interval",
                type: "line",
                data: points.map((item) => item.upper_90),
                symbol: "none",
                lineStyle: { color: "#a5bfb8", width: 1, type: "dashed" },
                tooltip: { show: false },
              },
              {
                name: "Actual revenue",
                type: "line",
                data: points.map((item) => item.actual),
                symbol: "none",
                lineStyle: { color: "#283e4c", width: 1.6 },
                itemStyle: { color: "#283e4c" },
              },
              {
                name: "Predicted revenue",
                type: "line",
                data: points.map((item) => item.predicted),
                symbol: "none",
                lineStyle: { color: "#368d7b", width: 2 },
                itemStyle: { color: "#368d7b" },
                markArea: holdout
                  ? {
                      silent: true,
                      itemStyle: { color: "rgba(213,166,90,.08)" },
                      data: [
                        [
                          { name: "Holdout", xAxis: holdout.week },
                          { xAxis: points.at(-1)?.week },
                        ],
                      ],
                      label: { color: "#9d7840", fontSize: 10 },
                    }
                  : undefined,
              },
            ],
      }}
    />
  );
}
