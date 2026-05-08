type PlausibleOptions = {
  props?: Record<string, string | number | boolean>;
};

declare global {
  interface Window {
    plausible?: (eventName: string, options?: PlausibleOptions) => void;
  }
}

export function trackEvent(
  eventName: string,
  props?: Record<string, string | number | boolean>,
): void {
  if (typeof window === "undefined") return;
  if (typeof window.plausible !== "function") return;
  window.plausible(eventName, props ? { props } : undefined);
}

export function trackOncePerSession(
  key: string,
  eventName: string,
  props?: Record<string, string | number | boolean>,
): void {
  if (typeof window === "undefined") return;
  const storageKey = `violawake.analytics.${key}`;
  try {
    if (window.sessionStorage.getItem(storageKey)) return;
    window.sessionStorage.setItem(storageKey, "1");
  } catch {
    // Analytics should never affect the product workflow.
  }
  trackEvent(eventName, props);
}
