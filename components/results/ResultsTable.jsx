import { useMemo } from 'react';
import { useTable, useSortBy, useFilters } from 'react-table';

export const ResultsTable = ({ data, onViewDetails }) => {
  const columns = useMemo(() => [
    {
      Header: 'Filename',
      accessor: 'filename',
      Cell: ({ value }) => (
        <span className="font-medium dark:text-white">{value}</span>
      )
    },
    {
      Header: 'Category',
      accessor: 'category'
    },
    {
      Header: 'Confidence',
      accessor: 'confidence',
      Cell: ({ value }) => (
        <ConfidenceCell value={value} />
      )
    },
    {
      Header: 'Status',
      accessor: 'status',
      Cell: ({ value }) => (
        <StatusBadge status={value} />
      )
    },
    {
      Header: 'Actions',
      Cell: ({ row }) => (
        <button
          onClick={() => onViewDetails(row.original)}
          className="text-blue-600 hover:text-blue-800 dark:text-blue-400 
            dark:hover:text-blue-200"
        >
          View Details
        </button>
      )
    }
  ], [onViewDetails]);

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow
  } = useTable(
    { columns, data },
    useFilters,
    useSortBy
  );

  return (
    <div className="overflow-x-auto">
      <table {...getTableProps()} className="w-full">
        {/* Table implementation */}
      </table>
    </div>
  );
}; 