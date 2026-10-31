/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#0B132B',
          800: '#1C2541',
          700: '#3A506B',
        },
        accent: {
          blue: '#48E5C2',
          cyan: '#5BC0BE',
          amber: '#F6AE2D',
          rose: '#F26419'
        }
      }
    },
  },
  plugins: [],
}
