use std::str::FromStr;

use polars_core::prelude::{
    DataFrame, ExplodeOptions, PolarsError, PolarsResult, polars_bail, polars_ensure,
};
#[cfg(feature = "csv")]
use polars_lazy::prelude::LazyCsvReader;
use polars_lazy::prelude::{IntoLazy, LazyFrame};
use polars_plan::utils::expr_to_leaf_column_names_iter;
use polars_utils::pl_path::PlRefPath;
use sqlparser::ast::{
    Expr as SQLExpr, FunctionArg as SQLFunctionArg, FunctionArgExpr as SQLFunctionArgExpr,
    Value as SQLValue, ValueWithSpan as SQLValueWithSpan,
};

use crate::SQLContext;
use crate::context::SubqueryShape;
use crate::functions::series_expr;
use crate::sql_expr::parse_sql_expr;

/// Table functions that are supported by Polars
#[allow(clippy::enum_variant_names)]
pub(crate) enum PolarsTableFunctions {
    /// SQL 'read_csv' function.
    /// ```sql
    /// SELECT * FROM read_csv('path/to/file.csv')
    /// ```
    #[cfg(feature = "csv")]
    ReadCsv,
    /// SQL 'read_parquet' function.
    /// ```sql
    /// SELECT * FROM read_parquet('path/to/file.parquet')
    /// ```
    #[cfg(feature = "parquet")]
    ReadParquet,
    /// SQL 'read_ipc' function.
    /// ```sql
    /// SELECT * FROM read_ipc('path/to/file.ipc')
    /// ```
    #[cfg(feature = "ipc")]
    ReadIpc,
    /// SQL 'read_json' function (*only ndjson is currently supported*).
    /// ```sql
    /// SELECT * FROM read_json('path/to/file.json')
    /// ```
    #[cfg(feature = "json")]
    ReadJson,
    /// SQL 'generate_series' function (stop included).
    /// ```sql
    /// SELECT * FROM generate_series(1, 10, 2)
    /// SELECT * FROM generate_series(DATE '2026-01-01', DATE '2026-01-31', INTERVAL '1 day')
    /// ```
    GenerateSeries,
    /// SQL 'range' function (stop not included).
    /// ```sql
    /// SELECT * FROM range(0, 10, 2)
    /// ```
    Range,
}

impl FromStr for PolarsTableFunctions {
    type Err = PolarsError;

    #[allow(unreachable_code)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            #[cfg(feature = "csv")]
            "read_csv" => PolarsTableFunctions::ReadCsv,
            #[cfg(feature = "parquet")]
            "read_parquet" => PolarsTableFunctions::ReadParquet,
            #[cfg(feature = "ipc")]
            "read_ipc" => PolarsTableFunctions::ReadIpc,
            #[cfg(feature = "json")]
            "read_json" => PolarsTableFunctions::ReadJson,
            "generate_series" => PolarsTableFunctions::GenerateSeries,
            "range" => PolarsTableFunctions::Range,
            _ => polars_bail!(SQLInterface: "'{}' is not a supported table function", s),
        })
    }
}

impl PolarsTableFunctions {
    #[allow(unreachable_code)]
    pub(crate) fn execute(
        &self,
        args: &[SQLFunctionArg],
        ctx: &mut SQLContext,
    ) -> PolarsResult<(String, LazyFrame)> {
        let (path, lf): (PlRefPath, LazyFrame) = match self {
            #[cfg(feature = "csv")]
            PolarsTableFunctions::ReadCsv => self.read_csv(args)?,
            #[cfg(feature = "parquet")]
            PolarsTableFunctions::ReadParquet => self.read_parquet(args)?,
            #[cfg(feature = "ipc")]
            PolarsTableFunctions::ReadIpc => self.read_ipc(args)?,
            #[cfg(feature = "json")]
            PolarsTableFunctions::ReadJson => self.read_ndjson(args)?,
            PolarsTableFunctions::GenerateSeries => return self.series(args, ctx, true),
            PolarsTableFunctions::Range => return self.series(args, ctx, false),
        };
        Ok((path.to_string(), lf))
    }

    fn series(
        &self,
        args: &[SQLFunctionArg],
        ctx: &mut SQLContext,
        include_end: bool,
    ) -> PolarsResult<(String, LazyFrame)> {
        let fname = if include_end {
            "generate_series"
        } else {
            "range"
        };
        let args = args
            .iter()
            .map(|arg| match arg {
                SQLFunctionArg::Unnamed(SQLFunctionArgExpr::Expr(e)) => Ok(e),
                _ => polars_bail!(SQLSyntax: "`{}` expects positional arguments; found {}", fname, arg),
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let mut expr = series_expr(fname, &args, include_end, |e| {
            let expr = parse_sql_expr(e, ctx, None)?;
            polars_ensure!(
                expr_to_leaf_column_names_iter(&expr).next().is_none(),
                SQLInterface: "`{}` in FROM takes only constant arguments; use UNNEST({}(...)) in the SELECT list for a series per row", fname, fname
            );
            Ok(expr)
        })?;
        let (lf, _) = ctx.process_subqueries(
            DataFrame::empty_with_height(1).lazy(),
            vec![&mut expr],
            SubqueryShape::Scalar,
        )?;
        let lf = lf.select([expr
            .explode(ExplodeOptions {
                empty_as_null: false,
                keep_nulls: false,
            })
            .alias(fname)]);
        Ok((fname.to_string(), lf))
    }

    #[cfg(feature = "csv")]
    fn read_csv(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_csv` expects a single file path; found {:?} arguments", args.len());

        use polars_lazy::frame::LazyFileListReader;
        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyCsvReader::new(path.clone())
            .with_try_parse_dates(true)
            .with_missing_is_null(true)
            .finish()?;
        Ok((path, lf))
    }

    #[cfg(feature = "parquet")]
    fn read_parquet(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_parquet` expects a single file path; found {:?} arguments", args.len());

        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyFrame::scan_parquet(path.clone(), Default::default())?;
        Ok((path, lf))
    }

    #[cfg(feature = "ipc")]
    fn read_ipc(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_ipc` expects a single file path; found {:?} arguments", args.len());

        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyFrame::scan_ipc(path.clone(), Default::default(), Default::default())?;
        Ok((path, lf))
    }
    #[cfg(feature = "json")]
    fn read_ndjson(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_ndjson` expects a single file path; found {:?} arguments", args.len());

        use polars_lazy::frame::LazyFileListReader;
        use polars_lazy::prelude::LazyJsonLineReader;

        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyJsonLineReader::new(path.clone()).finish()?;
        Ok((path, lf))
    }

    #[allow(dead_code)]
    fn get_file_path_from_arg(&self, arg: &SQLFunctionArg) -> PolarsResult<PlRefPath> {
        match arg {
            SQLFunctionArg::Unnamed(SQLFunctionArgExpr::Expr(SQLExpr::Value(
                SQLValueWithSpan {
                    value: SQLValue::SingleQuotedString(s),
                    ..
                },
            ))) => Ok(PlRefPath::new(s)),
            _ => polars_bail!(
                SQLSyntax:
                "expected a valid file path as a single-quoted string; found: {}", arg,
            ),
        }
    }
}

impl PolarsTableFunctions {
    // list sql names of all table functions
    pub(crate) fn keywords() -> &'static [&'static str] {
        &[
            #[cfg(feature = "csv")]
            "read_csv",
            #[cfg(feature = "parquet")]
            "read_parquet",
            #[cfg(feature = "ipc")]
            "read_ipc",
            #[cfg(feature = "json")]
            "read_json",
            "generate_series",
            "range",
        ]
    }
}
