# Trading Dashboard

The active application is the Next.js trading journal in [`monthly-trading-report/`](monthly-trading-report/).

## Local development

```bash
cd monthly-trading-report
npm ci
npm run dev
```

Local development uses JSON storage when `DATABASE_URL` is unset. See the [application README](monthly-trading-report/README.md) for local demo logins, configuration, features, and Vercel deployment instructions.

## Validation

Run from `monthly-trading-report/`:

```bash
npm run build
node --import tsx --test tests/*.test.ts tests/*.test.mjs tests/*.test.js
```

The retired Python/Streamlit dashboard has been removed. Historical data, screenshots, reference material, and standalone analysis scripts remain in the repository.
