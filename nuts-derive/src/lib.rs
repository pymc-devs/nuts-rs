extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    AngleBracketedGenericArguments, Data, DeriveInput, Fields, GenericParam, Ident, Lit, LitStr,
    PathArguments, Token, Type, TypePath,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

// Helper struct to parse `#[storable(dims(...))]` or #[storable(flattened)]
enum StorableAttr {
    Item(Vec<LitStr>),
    Flattened(),
    Ignore(),
}

impl Parse for StorableAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let metas = Punctuated::<syn::Meta, Token![,]>::parse_terminated(input)?;

        for meta in metas {
            match meta {
                syn::Meta::List(list) => {
                    if list.path.is_ident("dims") {
                        return Ok(StorableAttr::Item(
                            list.nested
                                .into_iter()
                                .map(|e| match e {
                                    syn::NestedMeta::Lit(Lit::Str(s)) => Ok(s),
                                    _ => Err(syn::Error::new_spanned(e, "Expected string literal")),
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        ));
                    }
                }
                syn::Meta::Path(path) => {
                    if path.is_ident("flatten") {
                        return Ok(StorableAttr::Flattened());
                    }
                    if path.is_ident("ignore") {
                        return Ok(StorableAttr::Ignore());
                    }
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        meta,
                        "Unsupported storable attribute. Expected `dims(...)` or `flatten`",
                    ));
                }
            }
        }

        Ok(StorableAttr::Item(vec![]))
    }
}

struct StorableBasicField {
    name: Ident,
    item_type: proc_macro2::TokenStream,
    is_vec: bool,
    is_option: bool,
    dims: Vec<LitStr>,
}

struct StorableInnerField {
    name: Ident,
    item_type: proc_macro2::TokenStream,
    is_option: bool,
}

enum StorableField {
    Basic(StorableBasicField),
    Inner(StorableInnerField),
    Generic(StorableInnerField),
}

// Check if a type is a generic type parameter
fn is_generic_param(ty: &Type, generics: &syn::Generics) -> bool {
    if let Type::Path(type_path) = ty {
        if type_path.path.segments.len() == 1 {
            let type_name = &type_path.path.segments.first().unwrap().ident;
            return generics.params.iter().any(|param| {
                if let GenericParam::Type(type_param) = param {
                    &type_param.ident == type_name
                } else {
                    false
                }
            });
        }
    }
    false
}

// Check if a type implements Storable trait based on bounds
fn has_storable_bound(ty: &Ident, generics: &syn::Generics) -> bool {
    for param in &generics.params {
        if let GenericParam::Type(type_param) = param {
            if &type_param.ident == ty {
                for bound in &type_param.bounds {
                    if let syn::TypeParamBound::Trait(trait_bound) = bound {
                        let path = &trait_bound.path;
                        if path.segments.len() == 1
                            && path.segments.first().unwrap().ident == "Storable"
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

#[proc_macro_derive(Storable, attributes(storable))]
pub fn storable_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let generics = &ast.generics;

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let impl_generics = if generics.params.is_empty() {
        quote! { <P: nuts_storable::HasDims> }
    } else {
        quote! { #impl_generics }
    };

    let fields = if let Data::Struct(s) = ast.data {
        if let Fields::Named(fields) = s.fields {
            fields.named
        } else {
            panic!("Storable can only be derived for structs with named fields");
        }
    } else {
        panic!("Storable can only be derived on structs");
    };

    let mut storable_fields = Vec::new();
    for field in fields {
        let field_name = field.ident.clone().unwrap();
        let ty = &field.ty;
        let ty_str = quote!(#ty).to_string();

        let attr = field
            .attrs
            .iter()
            .find(|a| a.path.is_ident("storable"))
            .map(|a| a.parse_args::<StorableAttr>().unwrap());

        if let Some(StorableAttr::Ignore()) = attr {
            continue; // Skip this field
        }

        let attr = attr.unwrap_or(StorableAttr::Item(vec![]));

        if let StorableAttr::Flattened() = attr {
            let path = if let Type::Path(TypePath { path: p, qself: _ }) = ty {
                p
            } else {
                panic!(
                    "Unsupported field type with flattened attribute: {}",
                    ty_str
                );
            };
            let item = if path.segments.first().unwrap().ident.to_string() == "Option" {
                if let PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                    args, ..
                }) = &path.segments.first().unwrap().arguments
                {
                    if let Some(arg) = args.first() {
                        let inner_type = quote!(#arg);
                        StorableField::Inner(StorableInnerField {
                            name: field_name.clone(),
                            item_type: inner_type,
                            is_option: true,
                        })
                    } else {
                        panic!("Invalid Option type for flattened field");
                    }
                } else {
                    panic!("Invalid Option type for flattened field");
                }
            } else {
                StorableField::Inner(StorableInnerField {
                    name: field_name.clone(),
                    item_type: path.into_token_stream(),
                    is_option: false,
                })
            };
            storable_fields.push(item);
            continue;
        }

        let dims = if let StorableAttr::Item(dims) = attr {
            dims
        } else {
            vec![]
        };

        // Check if the field is a generic type parameter
        if let Type::Path(type_path) = ty {
            if type_path.path.segments.len() == 1 {
                let type_name = &type_path.path.segments.first().unwrap().ident;

                // Check if this is a generic type parameter with Storable bound
                if is_generic_param(ty, generics) && has_storable_bound(type_name, generics) {
                    storable_fields.push(StorableField::Generic(StorableInnerField {
                        name: field_name,
                        item_type: quote!(#type_name),
                        is_option: false,
                    }));
                    continue;
                }

                // Check if this is Option<T> where T is a generic type parameter
                if type_name == "Option" {
                    if let PathArguments::AngleBracketed(args) =
                        &type_path.path.segments.first().unwrap().arguments
                    {
                        if let Some(arg) = args.args.first() {
                            if let syn::GenericArgument::Type(inner_ty) = arg {
                                if let Type::Path(inner_path) = inner_ty {
                                    if inner_path.path.segments.len() == 1 {
                                        let inner_name =
                                            &inner_path.path.segments.first().unwrap().ident;
                                        if is_generic_param(inner_ty, generics)
                                            && has_storable_bound(inner_name, generics)
                                        {
                                            storable_fields.push(StorableField::Generic(
                                                StorableInnerField {
                                                    name: field_name,
                                                    item_type: quote!(#inner_name),
                                                    is_option: true,
                                                },
                                            ));
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let item = match ty_str.as_str() {
            "u64" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::U64 },
                is_vec: false,
                is_option: false,
                dims,
            }),
            "i64" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::I64 },
                is_vec: false,
                is_option: false,
                dims,
            }),
            "f64" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::F64 },
                is_vec: false,
                is_option: false,
                dims,
            }),
            "f32" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::F32 },
                is_vec: false,
                is_option: false,
                dims,
            }),
            "bool" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::Bool },
                is_vec: false,
                is_option: false,
                dims,
            }),
            "Option < u64 >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::U64 },
                is_vec: false,
                is_option: true,
                dims,
            }),
            "Option < i64 >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::I64 },
                is_vec: false,
                is_option: true,
                dims,
            }),
            "Option < f64 >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::F64 },
                is_vec: false,
                is_option: true,
                dims,
            }),
            "Option < f32 >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::F32 },
                is_vec: false,
                is_option: true,
                dims,
            }),
            "Option < bool >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::Bool },
                is_vec: false,
                is_option: true,
                dims,
            }),
            "Vec < u64 >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::U64 },
                is_vec: true,
                is_option: false,
                dims,
            }),
            "Vec < i64 >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::I64 },
                is_vec: true,
                is_option: false,
                dims,
            }),
            "Vec < f64 >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::F64 },
                is_vec: true,
                is_option: false,
                dims,
            }),
            "Vec < f32 >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::F32 },
                is_vec: true,
                is_option: false,
                dims,
            }),
            "Vec < bool >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::Bool },
                is_vec: true,
                is_option: false,
                dims,
            }),
            "Option < Vec < u64 > >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::U64 },
                is_vec: true,
                is_option: true,
                dims,
            }),
            "Option < Vec < i64 > >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::I64 },
                is_vec: true,
                is_option: true,
                dims,
            }),
            "Option < Vec < f64 > >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::F64 },
                is_vec: true,
                is_option: true,
                dims,
            }),
            "Option < Vec < f32 > >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::F32 },
                is_vec: true,
                is_option: true,
                dims,
            }),
            "Option< Vec < bool > >" => StorableField::Basic(StorableBasicField {
                name: field_name.clone(),
                item_type: quote! { nuts_storable::ItemType::Bool },
                is_vec: true,
                is_option: true,
                dims,
            }),
            _ => {
                // Attempt to handle complex generic types that are still Storable
                if let Type::Path(type_path) = ty {
                    // Check if it's a type that has the Storable trait
                    let type_token = quote!(#type_path);
                    storable_fields.push(StorableField::Inner(StorableInnerField {
                        name: field_name.clone(),
                        item_type: type_token,
                        is_option: false,
                    }));
                    continue;
                } else {
                    panic!("Unsupported field type: {}", ty_str);
                }
            }
        };
        storable_fields.push(item);
    }

    let names_exprs = storable_fields.iter().map(|f| match f {
        StorableField::Basic(field) => {
            let name = field.name.to_string();
            quote! { vec![#name] }
        }
        StorableField::Inner(field) => {
            let item_type = &field.item_type;
            quote! { #item_type::names(parent) }
        }
        StorableField::Generic(field) => {
            let name = field.name.to_string();
            if field.is_option {
                quote! { vec![#name] }
            } else {
                let item_type = &field.item_type;
                quote! { #item_type::names(parent) }
            }
        }
    });

    let names_fn = quote! {
        fn names(parent: &P) -> Vec<&str> {
            let mut names = Vec::new();
            #(names.extend(#names_exprs);)*
            names
        }
    };

    let item_type_arms = storable_fields.iter().map(|f| match f {
        StorableField::Basic(field) => {
            let name_str = field.name.to_string();
            let item_type = &field.item_type;
            quote! { #name_str => #item_type, }
        }
        StorableField::Inner(field) => {
            let item_type = &field.item_type;
            quote! { name if #item_type::names(parent).contains(&name) => #item_type::item_type(parent, name), }
        }
        StorableField::Generic(field) => {
            let name_str = field.name.to_string();
            let item_type = &field.item_type;
            if field.is_option {
                quote! { #name_str => nuts_storable::ItemType::Generic, }
            } else {
                quote! { name if #item_type::names(parent).contains(&name) => #item_type::item_type(parent, name), }
            }
        }
    });

    let item_type_fn = quote! {
        fn item_type(parent: &P, item: &str) -> nuts_storable::ItemType {
            match item {
                #(#item_type_arms)*
                _ => { panic!("Unknown item: {}", item); }
            }
        }
    };

    let dims_arms = storable_fields.iter().map(|f| match f {
        StorableField::Basic(field) => {
            let name_str = field.name.to_string();
            let dims = &field.dims;
            quote! { #name_str => vec![#(#dims),*], }
        }
        StorableField::Inner(field) => {
            let item_type = &field.item_type;
            quote! { name if #item_type::names(parent).contains(&name) => #item_type::dims(parent, name), }
        }
        StorableField::Generic(field) => {
            let name_str = field.name.to_string();
            let item_type = &field.item_type;
            if field.is_option {
                quote! { #name_str => vec![], }
            } else {
                quote! { name if #item_type::names(parent).contains(&name) => #item_type::dims(parent, name), }
            }
        }
    });

    let dims_fn = quote! {
        fn dims<'a>(parent: &'a P, item: &str) -> Vec<&'a str> {
            match item {
                #(#dims_arms)*
                _ => { panic!("Unknown item: {}", item); }
            }
        }
    };

    let get_all_exprs = storable_fields.iter().map(|f| match f {
        StorableField::Basic(field) => {
            let name = &field.name;
            let name_str = name.to_string();
            let value_expr = if field.is_option {
                if field.is_vec {
                    quote! { self.#name.as_ref().map(|v| nuts_storable::Value::from(v.clone())) }
                } else {
                    quote! { self.#name.map(nuts_storable::Value::from) }
                }
            } else {
                quote! { Some(nuts_storable::Value::from(self.#name.clone())) }
            };
            quote! { result.push((#name_str, #value_expr)); }
        }
        StorableField::Inner(field) => {
            let name = &field.name;
            if field.is_option {
                quote! {
                    if let Some(inner) = &self.#name {
                        result.extend(inner.get_all(parent));
                    }
                }
            } else {
                quote! { result.extend(self.#name.get_all(parent)); }
            }
        }
        StorableField::Generic(field) => {
            let name = &field.name;
            if field.is_option {
                quote! {
                    if let Some(inner) = &self.#name {
                        result.push((#name.to_string().as_str(), Some(nuts_storable::Value::Generic(Box::new(inner.clone())))));
                    } else {
                        result.push((#name.to_string().as_str(), None));
                    }
                }
            } else {
                quote! { result.extend(self.#name.get_all(parent)); }
            }
        }
    });

    let get_all_fn = quote! {
        fn get_all(&self, parent: &P) -> Vec<(&str, Option<nuts_storable::Value>)> {
            let mut result = Vec::with_capacity(Self::names(parent).len());
            #(#get_all_exprs)*
            result
        }
    };

    let r#gen = quote! {
        impl #impl_generics nuts_storable::Storable<P> for #name #ty_generics #where_clause {
            #names_fn
            #item_type_fn
            #dims_fn
            #get_all_fn
        }
    };

    r#gen.into()
}
