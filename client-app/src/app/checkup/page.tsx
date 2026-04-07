"use client";

import React, { useMemo, useState, useEffect } from "react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

import Image from "next/image";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  type ChartOptions,
  type ChartData,
  type TooltipItem,
} from "chart.js";
import { Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

/* ---------------------------
   Validation schema (Zod) - numeric category codes so FormSchema matches API
   --------------------------- */
const schema = z.object({
  age: z.number().min(1).max(120),
  gender: z.enum(["M", "F"]),
  chestpain: z.number().int().min(1).max(4),
  restingBP: z.number().min(30).max(300),
  serumcholestrol: z.number().min(50).max(1000),
  fastingbloodsugar: z.number().int().min(0).max(1),
  restingrelectro: z.number().int().min(0).max(2),
  maxheartrate: z.number().min(30).max(300),
  exerciseangia: z.number().int().min(0).max(1),
  oldpeak: z.number().min(0).max(20),
  slope: z.number().int().min(1).max(3),
  noofmajorvessels: z.number().int().min(0).max(3),
});
type FormSchema = z.infer<typeof schema>;

const chestpainOptions = [
  { label: "Typical angina", value: 1 },
  { label: "Atypical angina", value: 2 },
  { label: "Non-anginal pain", value: 3 },
  { label: "Asymptomatic", value: 4 },
];

const slopeOptions = [
  { label: "Upsloping", value: 1 },
  { label: "Flat", value: 2 },
  { label: "Downsloping", value: 3 },
];

/* ---------------------------
   Default payload used for quick tests (typed)
   --------------------------- */
const DEFAULT_PAYLOAD: FormSchema = {
  age: 54,
  gender: "M",
  chestpain: 3,
  restingBP: 140,
  serumcholestrol: 240,
  fastingbloodsugar: 0,
  restingrelectro: 1,
  maxheartrate: 150,
  exerciseangia: 0,
  oldpeak: 1.2,
  slope: 2,
  noofmajorvessels: 0,
};

/* ---------------------------
   Page Component
   --------------------------- */
export default function PredictPage() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
  } = useForm<FormSchema>({
    resolver: zodResolver(schema),
    defaultValues: DEFAULT_PAYLOAD,
  });

  const [prediction, setPrediction] = useState<number | null>(null);
  const [probability, setProbability] = useState<number | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);

  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [explainResult, setExplainResult] = useState<{
    prediction?: number;
    probability?: number;
    contributions?: Record<string, number>;
    aggregated?: Record<string, number>;
    top_features?: { feature: string; impact: number }[];
    base_value?: number | null;
  } | null>(null);

  const [loading, setLoading] = useState(false);
  const [showForm, setShowForm] = useState(true);

  // theme colors read from CSS variables (client-only)
  const [posColor, setPosColor] = useState<string>("");
  const [negColor, setNegColor] = useState<string>("");
  useEffect(() => {
    const s = getComputedStyle(document.documentElement);
    const p = s.getPropertyValue("--color-chart-1") || "#4f46e5";
    const n = s.getPropertyValue("--color-chart-2") || "#ef4444";
    setPosColor(p.trim() || "#4f46e5");
    setNegColor(n.trim() || "#ef4444");
  }, []);

  // typed API calls
  async function callPredict(payload: FormSchema) {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return (await res.json()) as { prediction: number; probability: number };
  }

  async function callExplain(payload: FormSchema) {
    const res = await fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return (await res.json()) as {
      prediction: number;
      probability: number;
      contributions?: Record<string, number>;
      aggregated?: Record<string, number>;
      top_features?: { feature: string; impact: number }[];
      base_value?: number | null;
    };
  }

  // run test with DEFAULT_PAYLOAD (or provided) — used on mount and after form submit
  async function runTest(payload: FormSchema = DEFAULT_PAYLOAD) {
    setLoading(true);
    setServerError(null);
    setExplainError(null);
    setExplainResult(null);
    setPrediction(null);
    setProbability(null);

    try {
      setExplainLoading(true);
      const e = await callExplain(payload);
      
      // Set prediction and probability from the consolidated /explain response
      setPrediction(Number(e.prediction));
      const prob = Number(e.probability);
      setProbability(prob > 1 ? prob : prob * 100);

      setExplainResult({
        contributions: e.contributions ?? undefined,
        aggregated: e.aggregated ?? undefined,
        top_features: e.top_features ?? undefined,
        base_value: e.base_value ?? null,
      });
    } catch (err: unknown) {
      console.error("Diagnostic analysis failed", err);
      setServerError(
        String(err instanceof Error ? err.message : "Diagnostic analysis failed")
      );
    } finally {
      setLoading(false);
      setExplainLoading(false);
    }
  }

  // initial test on mount
  React.useEffect(() => {
    void runTest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // compute chart-ready dataset from explainResult.top_features
  const chartData = useMemo(() => {
    // Show ALL features from aggregated if available, falling back to top_features
    const rawData = explainResult?.aggregated 
      ? Object.entries(explainResult.aggregated).map(([feature, impact]) => ({ feature, impact }))
      : (explainResult?.top_features ?? []);
      
    // sort by absolute impact desc, put largest first
    const sorted = [...rawData].sort(
      (a, b) => Math.abs(b.impact) - Math.abs(a.impact)
    );
    const labels = sorted.map((t) => t.feature.replace(/_/g, " "));
    const values = sorted.map((t) => Number(t.impact));
    const backgroundColors = sorted.map((t) =>
      t.impact >= 0 ? posColor : negColor
    );
    return { labels, values, backgroundColors };
  }, [explainResult, posColor, negColor]);

  // Chart.js options: horizontal bar (typed)
  const chartOptions: ChartOptions<"bar"> = useMemo(
    () => ({
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            // no 'any' here
            label: (ctx: TooltipItem<"bar">) => {
              // ctx.raw in our dataset is a number; but handle object just in case.
              const raw = ctx.raw as number | { x?: number; y?: number };
              const value =
                typeof raw === "number"
                  ? raw
                  : typeof raw?.x === "number"
                    ? raw.x
                    : typeof raw?.y === "number"
                      ? raw.y
                      : Number(raw);
              return `Impact: ${Number(value).toFixed(3)}`;
            },
          },
        },
        title: {
          display: true,
          text: "Top feature contributions",
        },
      },
      scales: {
        x: {
          ticks: {
            callback: (val: string | number) =>
              typeof val === "number" ? val.toFixed(2) : val,
          },
        },
        y: {
          ticks: { autoSkip: false },
        },
      },
    }),
    []
  );

  // Typed dataset (no 'as any')
  const chartDataset: ChartData<"bar", number[], string> = useMemo(
    () => ({
      labels: chartData.labels,
      datasets: [
        {
          label: "Impact",
          data: chartData.values,
          backgroundColor: chartData.backgroundColors,
          borderRadius: 6,
          barThickness: 18,
        },
      ],
    }),
    [chartData]
  );

  // form submit handler — run and close form
  async function onSubmit(values: FormSchema) {
    await runTest(values);
    setShowForm(false);
  }

  return (
    <div
      className="min-h-screen relative"
      style={{
        background: "var(--color-background)",
        color: "var(--color-foreground)",
        padding: 20,
      }}
    >
      <div className=" mx-auto z-10 relative bg-transparent">
        <div className="flex items-center justify-between mb-4 z-10">
          <h1 className="text-2xl font-bold">
            Heart Disease — Predict & Explain
          </h1>
          {!showForm && (
            <div style={{ display: "flex", gap: 8 }}>
              <Button
                onClick={() => setShowForm(true)}
                disabled={loading || explainLoading}
              >
                {loading || explainLoading ? "Busy…" : "Test again"}
              </Button>
            </div>
          )}
        </div>

        {/* FORM (toggle) */}
        {showForm && (
          <Card className="mb-6 max-w-6xl p-4 z-10 bg-transparent backdrop-blur-sm">
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4 z-10">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <Label htmlFor="age" className="p-2">
                    Age
                  </Label>
                  <Input
                    id="age"
                    type="number"
                    step="1"
                    {...register("age", { valueAsNumber: true })}
                  />
                  {errors.age && (
                    <div className="text-xs text-red-600 mt-1">
                      {errors.age.message}
                    </div>
                  )}
                </div>

                <div>
                  <Label htmlFor="gender" className="p-2">
                    Gender
                  </Label>
                  <select
                    id="gender"
                    {...register("gender")}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                    style={{
                      background: "var(--color-card)",
                      color: "var(--color-card-foreground)",
                      borderColor: "var(--color-border)",
                    }}
                  >
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                  </select>
                  {errors.gender && (
                    <div className="text-xs text-red-600 mt-1">
                      {errors.gender.message}
                    </div>
                  )}
                </div>

                <div>
                  <Label htmlFor="chestpain" className="p-2">
                    Chest Pain
                  </Label>
                  <select
                    id="chestpain"
                    {...register("chestpain", { valueAsNumber: true })}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                    style={{
                      background: "var(--color-card)",
                      color: "var(--color-card-foreground)",
                      borderColor: "var(--color-border)",
                    }}
                  >
                    {chestpainOptions.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <Label htmlFor="restingBP" className="p-2">
                    Resting BP
                  </Label>
                  <Input
                    id="restingBP"
                    type="number"
                    step="1"
                    {...register("restingBP", { valueAsNumber: true })}
                  />
                </div>

                <div>
                  <Label htmlFor="serumcholestrol">Cholesterol</Label>
                  <Input
                    id="serumcholestrol"
                    type="number"
                    step="1"
                    {...register("serumcholestrol", { valueAsNumber: true })}
                  />
                </div>

                <div>
                  <Label htmlFor="fastingbloodsugar" className="p-2">
                    Fasting Blood Sugar
                  </Label>
                  <select
                    id="fastingbloodsugar"
                    {...register("fastingbloodsugar", { valueAsNumber: true })}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                    style={{
                      background: "var(--color-card)",
                      color: "var(--color-card-foreground)",
                      borderColor: "var(--color-border)",
                    }}
                  >
                    <option value={0}>{"No (<=120)"}</option>
                    <option value={1}>Yes (&gt;120)</option>
                  </select>
                </div>

                <div>
                  <Label htmlFor="restingrelectro" className="p-2">
                    Resting ECG
                  </Label>
                  <select
                    id="restingrelectro"
                    {...register("restingrelectro", { valueAsNumber: true })}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                    style={{
                      background: "var(--color-card)",
                      color: "var(--color-card-foreground)",
                      borderColor: "var(--color-border)",
                    }}
                  >
                    <option value={0}>Normal</option>
                    <option value={1}>ST-T abnormality</option>
                    <option value={2}>Left ventricular hypertrophy</option>
                  </select>
                </div>

                <div>
                  <Label htmlFor="maxheartrate" className="p-2">
                    Max Heart Rate
                  </Label>
                  <Input
                    id="maxheartrate"
                    type="number"
                    step="1"
                    {...register("maxheartrate", { valueAsNumber: true })}
                  />
                </div>

                <div>
                  <Label htmlFor="exerciseangia" className="p-2">
                    Exercise Angina
                  </Label>
                  <select
                    id="exerciseangia"
                    {...register("exerciseangia", { valueAsNumber: true })}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                    style={{
                      background: "var(--color-card)",
                      color: "var(--color-card-foreground)",
                      borderColor: "var(--color-border)",
                    }}
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>

                <div>
                  <Label htmlFor="oldpeak" className="p-2">
                    Oldpeak (ST depression)
                  </Label>
                  <Input
                    id="oldpeak"
                    type="number"
                    step="0.1"
                    {...register("oldpeak", { valueAsNumber: true })}
                  />
                </div>

                <div>
                  <Label htmlFor="slope" className="p-2">
                    Slope
                  </Label>
                  <select
                    id="slope"
                    {...register("slope", { valueAsNumber: true })}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                    style={{
                      background: "var(--color-card)",
                      color: "var(--color-card-foreground)",
                      borderColor: "var(--color-border)",
                    }}
                  >
                    {slopeOptions.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <Label htmlFor="noofmajorvessels" className="p-2">
                    No. major vessels
                  </Label>
                  <select
                    id="noofmajorvessels"
                    {...register("noofmajorvessels", { valueAsNumber: true })}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                    style={{
                      background: "var(--color-card)",
                      color: "var(--color-card-foreground)",
                      borderColor: "var(--color-border)",
                    }}
                  >
                    <option value={0}>0</option>
                    <option value={1}>1</option>
                    <option value={2}>2</option>
                    <option value={3}>3</option>
                  </select>
                </div>
              </div>

              {serverError && (
                <div
                  className="text-sm p-2 rounded"
                  style={{
                    background: "var(--destructive)",
                    color: "var(--primary-foreground)",
                  }}
                >
                  {serverError}
                </div>
              )}

              <div className="flex gap-3">
                <Button type="submit" disabled={isSubmitting}>
                  {isSubmitting ? "Predicting..." : "Predict"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => {
                    reset();
                    setShowForm(false);
                  }}
                >
                  Cancel
                </Button>
              </div>
            </form>
          </Card>
        )}

        {/* Prediction & Explanation tiles (chart) */}
        {!showForm && (
          <div className="w-full grid grid-cols-1 gap-4">
            {/* 2x2 Dashboard grid (responsive) */}
            <div className="mt-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Tile 1: Prediction (top-left) */}
                <Card className="p-4 h-max">
                  <div className="flex items-center justify-between">
                    <div>
                      <div
                        className="text-sm"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        Prediction
                      </div>
                      <div className="text-lg font-semibold">
                        {prediction === null
                          ? "—"
                          : prediction === 1
                            ? "Heart disease — Positive"
                            : "No heart disease — Negative"}
                      </div>
                    </div>
                    <div className="text-right">
                      <div
                        className="text-sm"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        Status
                      </div>
                      <div className="text-lg font-semibold">
                        {prediction === null
                          ? "Waiting"
                          : prediction === 1
                            ? "High risk"
                            : "Low risk"}
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Tile 2: Probability (top-right) */}
                <Card className="p-4 h-max">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                    <div>
                      <div
                        className="text-sm"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        Probability
                      </div>
                      <div className="text-lg font-semibold">
                        {(probability ?? 0).toFixed(2)}%
                      </div>
                    </div>

                    {/* optional small visual / ring */}
                    <div className="flex items-center justify-center">
                      <div
                        aria-hidden
                        style={{
                          width: 56,
                          height: 56,
                          borderRadius: 999,
                          display: "grid",
                          placeItems: "center",
                          background:
                            "linear-gradient(180deg,var(--color-card) 0%, rgba(255,255,255,0.02) 100%)",
                          border: "1px solid var(--color-border)",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 12,
                            color: "var(--muted-foreground)",
                          }}
                        >
                          conf
                        </div>
                        <div style={{ fontWeight: 700 }}>
                          {Math.round(probability ?? 0)}%
                        </div>
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Tile 3: Top features (bottom-left) */}
                <Card className="p-4 h-max">
                  <div>
                    <div
                      className="text-sm"
                      style={{ color: "var(--muted-foreground)" }}
                    >
                      Top features
                    </div>
                    {explainResult?.top_features &&
                    explainResult.top_features.length > 0 ? (
                      <ul className="mt-3 space-y-2">
                        {explainResult.top_features.map((t, i) => (
                          <li
                            key={i}
                            className="flex items-center justify-between"
                          >
                            <div className="text-sm truncate">
                              {t.feature.replace(/_/g, " ")}
                            </div>
                            <div
                              className="text-sm font-medium"
                              style={{
                                color:
                                  t.impact > 0
                                    ? "var(--color-chart-1)"
                                    : "var(--color-chart-2)",
                              }}
                            >
                              {t.impact > 0 ? "+" : ""}
                              {t.impact.toFixed(3)}
                            </div>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div
                        className="mt-3 text-sm"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        No explanation available — run a prediction.
                      </div>
                    )}
                  </div>
                </Card>

                {/* Tile 4: Aggregated contributions (bottom-right) */}
                <Card className="p-4 h-max">
                  <div>
                    <div
                      className="text-sm"
                      style={{ color: "var(--muted-foreground)" }}
                    >
                      Aggregated contributions
                    </div>

                    {explainResult?.aggregated ? (
                      <div className="mt-3">
                        <dl className="grid gap-2">
                          {Object.entries(explainResult.aggregated)
                            .map(([k, v]) => (
                              <div
                                key={k}
                                className="flex items-center justify-between text-sm"
                              >
                                <dt className="truncate">
                                  {k.replace(/_/g, " ")}
                                </dt>
                                <dd
                                  style={{
                                    color:
                                      (v as number) > 0
                                        ? "var(--color-chart-1)"
                                        : "var(--color-chart-2)",
                                    fontWeight: 600,
                                  }}
                                >
                                  {(v as number) > 0 ? "+" : ""}
                                  {(v as number).toFixed(3)}
                                </dd>
                              </div>
                            ))}
                        </dl>
                      </div>
                    ) : (
                      <div
                        className="mt-3 text-sm"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        No aggregated data yet.
                      </div>
                    )}
                  </div>
                </Card>
              </div>
            </div>

            <Card className="p-4 w-full">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-base font-semibold">Explanation</h3>
                <div
                  className="text-sm"
                  style={{ color: "var(--muted-foreground)" }}
                >
                  {explainLoading ? "Computing explanation…" : ""}
                </div>
              </div>

              {explainError && (
                <div
                  className="mb-3 text-sm p-2 rounded"
                  style={{
                    background: "var(--destructive)",
                    color: "var(--primary-foreground)",
                  }}
                >
                  {explainError}
                </div>
              )}

              {/* Chart area */}
              <div style={{ minHeight: 220 }}>
                {!explainResult ||
                !explainResult.top_features ||
                explainResult.top_features.length === 0 ? (
                  <div style={{ color: "var(--muted-foreground)" }}>
                    Submit a prediction to see feature contributions.
                  </div>
                ) : (
                  <div
                    style={{
                      height: Math.max(240, chartData.labels.length * 40),
                    }}
                  >
                    <Bar options={chartOptions} data={chartDataset} />
                  </div>
                )}
              </div>

              {explainResult?.aggregated && (
                <details className="mt-4">
                  <summary className="cursor-pointer">
                    Show aggregated contributions
                  </summary>
                  <pre
                    style={{
                      marginTop: 8,
                      padding: 8,
                      background: "var(--color-popover)",
                      color: "var(--color-popover-foreground)",
                      borderRadius: 6,
                      maxHeight: 220,
                      overflow: "auto",
                    }}
                  >
                    {JSON.stringify(explainResult.aggregated, null, 2)}
                  </pre>
                </details>
              )}
            </Card>
          </div>
        )}
      </div>
      {showForm && (
        <Image
          src={"/heart-doctor-icon.png"}
          width={400}
          height={1200}
          alt="bg-icon"
          className="absolute w-[30rem] right-10 bottom-6 z-[0]"
        />
      )}
    </div>
  );
}
