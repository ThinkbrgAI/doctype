import { DarkModeToggle } from './DarkModeToggle';

export const Header = () => (
  <div className="flex justify-between items-center mb-8">
    <div>
      <h1 className="text-3xl font-bold mb-2 dark:text-white">
        Document Classification System
      </h1>
      <p className="text-gray-600 dark:text-gray-400">
        Upload and classify your construction documents automatically
      </p>
    </div>
    <DarkModeToggle />
  </div>
); 